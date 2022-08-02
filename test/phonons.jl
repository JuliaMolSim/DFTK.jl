# TODO: Temporary, explanations too scarce. To be changed with proper phonon computations.
using Test
using DFTK
using LinearAlgebra
using ForwardDiff
using StaticArrays

@testset "Phonons" begin

# Convert back and forth between Vec3 and columnwise matrix
fold(x)   = hcat(x...)
unfold(x) = Vec3.(eachcol(x))

function prepare_system(; n_scell=1)
    positions = [[0.0, 0.0, 0.0]]
    for i in 1:n_scell-1
        push!(positions, i * ones(3) / n_scell)
    end

    a = 5. * length(positions)
    lattice = a * [[1 0 0.]; [0 0 0.]; [0 0 0.]]

    Linv = DFTK.compute_inverse_lattice(lattice)
    directions = [[Linv * [i==j, 0.0, 0.0] for i in 1:n_scell] for j in 1:n_scell]

    params = Dict((:X, :X) => (; ε=1, σ=a / length(positions) /2^(1/6)))
    V(x, p) = 4*p.ε * ((p.σ/x)^12 - (p.σ/x)^6)

    (; positions, lattice, directions, params, V)
end

# Compute phonons for a one-dimensional pairwise potential for a set of `q = 0` using
# supercell method
function test_supercell_q0(; n_scell=1, max_radius=1e3)
    case = prepare_system(; n_scell)
    Linv  = DFTK.compute_inverse_lattice(case.lattice)
    n_atoms = length(case.positions)

    directions = [reshape(vcat([[i==j, 0.0, 0.0] for i in 1:n_atoms]...), 3, :)
                  for j in 1:n_atoms]

    Φ = zeros(length(directions), n_atoms)
    for (i, direction) in enumerate(directions)
        Φ[i, :] = - ForwardDiff.derivative(0.0) do ε
            new_positions = unfold(fold(case.positions) .+ ε .* Linv * direction)
            forces = zeros(Vec3{complex(eltype(ε))}, n_atoms)
            DFTK.energy_pairwise(case.lattice, fill(:X, n_atoms),
                                 new_positions, case.V, case.params;
                                 forces, max_radius)
            [(Linv * f)[1] for f in forces]
        end
    end
    sqrt.(abs.(eigvals(Φ)))
end

# Compute phonons for a one-dimensional pairwise potential for a set of `q`-points
function test_ph_disp(; n_scell=1, max_radius=1e3, n_points=2)
    case = prepare_system(; n_scell)
    Linv  = DFTK.compute_inverse_lattice(case.lattice)
    n_atoms = length(case.positions)

    function pairwise_ph(q, d; forces=nothing)
        DFTK.energy_pairwise(case.lattice, fill(:X, n_atoms),
                             case.positions, case.V, case.params;
                             q=[q, 0, 0], ph_disp=d, forces, max_radius)
    end

    ph_bands = []
    for q in range(-1/2, 1/2, length=n_points+1)
        as = ComplexF64[]
        for d in case.directions
            res = -ForwardDiff.derivative(0.0) do ε
                forces = zeros(Vec3{complex(eltype(ε))}, n_atoms)
                pairwise_ph(q, ε*d; forces)
                [Linv' * f for f in forces]
            end
            [push!(as, r[1]) for r in res]
        end
        M = reshape(as, n_atoms, :)
        @assert norm(imag.(eigvals(M))) < 1e-8
        push!(ph_bands, sqrt.(abs.(real(eigvals(M)))))
    end
    ph_bands
end

@testset "Phonon consistency" begin
    max_radius = 1e3
    tolerance = 1e-6
    n_points = 10

    ph_bands = [test_ph_disp(; n_scell, max_radius, n_points) for n_scell in 1:3]

    # Recover the same extremum for the system whatever case we test
    for n_scell in 2:3
        @test minimum(fold(ph_bands[1])) ≈ minimum(fold(ph_bands[n_scell])) atol=tolerance
        @test maximum(fold(ph_bands[1])) ≈ maximum(fold(ph_bands[n_scell])) atol=tolerance
    end

    # Test consistency between supercell method at `q = 0` and direct `q`-points computations
    for n_scell in 1:3
        r_q0 = test_supercell_q0(; n_scell, max_radius)
        @assert length(r_q0) == n_scell
        ph_band_q0 = ph_bands[n_scell][n_points÷2+1]
        @test norm(r_q0 - ph_band_q0) < tolerance
    end
end

end
