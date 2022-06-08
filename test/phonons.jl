using Test
using DFTK
using LinearAlgebra
using ForwardDiff
using StaticArrays

# Convert back and forth between Vec3 and columnwise matrix
fold(x)   = hcat(x...)
unfold(x) = Vec3.(eachcol(x))

function prepare_system(; n_scell=1)
    positions = [[0.,0,0]]
    for i in 1:n_scell-1
        push!(positions, i*ones(3)/n_scell)
    end

    a = 5. * length(positions)
    lattice = a * [[1 0 0.]; [0 0 0.]; [0 0 0.]]

    s = DFTK.compute_inverse_lattice(lattice)
    directions = [[s * [i==j,0,0] for i in 1:n_scell] for j in 1:n_scell]

    params = Dict((:X, :X) => (; ε=1, σ=a / length(positions) /2^(1/6)))
    V(x, p) = 4*p.ε * ((p.σ/x)^12 - (p.σ/x)^6)

    (positions=positions, lattice=lattice, directions=directions, params=params, V=V)
end

# Compute phonons for a one-dimensional pairwise potential for a set of `q = 0` using
# supercell method
function test_supercell_q0(; n_scell=1, max_radius=1e3)
    blob = prepare_system(; n_scell)
    positions = blob.positions
    lattice = blob.lattice
    directions = blob.directions
    params = blob.params
    V = blob.V

    s = DFTK.compute_inverse_lattice(lattice)
    n_atoms = length(positions)

    directions = [reshape(vcat([[i==j, 0.0, 0.0] for i in 1:n_atoms]...), 3, :) for j in 1:n_atoms]

    Φ = Array{eltype(positions[1])}(undef, length(directions), n_atoms)
    for (i, direction) in enumerate(directions)
        Φ[i, :] = - ForwardDiff.derivative(0.0) do ε
            new_positions = unfold(fold(positions) .+ ε .* s * direction)
            forces = zeros(Vec3{complex(eltype(ε))}, length(positions))
            DFTK.energy_pairwise(lattice, [:X for _ in positions], new_positions, V, params;
                                 forces, max_radius)
            [(s * f)[1] for f in forces]
        end
    end
    sqrt.(abs.(eigvals(Φ)))
end

# Compute phonons for a one-dimensional pairwise potential for a set of `q`-points
function test_ph_disp(; n_scell=1, max_radius=1e3, n_points=2)
    blob = prepare_system(; n_scell)
    positions = blob.positions
    lattice = blob.lattice
    directions = blob.directions
    params = blob.params
    V = blob.V

    pairwise_ph = (q, d; forces=nothing) ->
                     DFTK.energy_pairwise(lattice, [:X for _ in positions],
                                          positions, V, params; q=[q, 0, 0],
                                          ph_disp=d, forces, max_radius)

    ph_bands = []
    qs = -1/2:1/n_points:1/2
    for q in qs
        as = ComplexF64[]
        for d in directions
            res = -ForwardDiff.derivative(0.0) do ε
                forces = zeros(Vec3{complex(eltype(ε))}, length(positions))
                pairwise_ph(q, ε*d; forces)
                [DFTK.compute_inverse_lattice(lattice)' *  f for f in forces]
            end
            [push!(as, r[1]) for r in res]
        end
        M = reshape(as, length(positions), :)
        @assert ≈(norm(imag.(eigvals(M))), 0.0, atol=1e-8)
        push!(ph_bands, sqrt.(abs.(real(eigvals(M)))))
    end
    return ph_bands
end

@testset "Phonon consistency" begin
    max_radius = 1e3
    tolerance = 1e-6
    n_points = 10

    ph_bands = []
    for n_scell in [1, 2, 3]
        push!(ph_bands, test_ph_disp(; n_scell, max_radius, n_points))
    end

    # Recover the same extremum for the system whatever case we test
    for n_scell in [2, 3]
        @test ≈(minimum(fold(ph_bands[1])), minimum(fold(ph_bands[n_scell])), atol=tolerance)
        @test ≈(maximum(fold(ph_bands[1])), maximum(fold(ph_bands[n_scell])), atol=tolerance)
    end

    # Test consistency between supercell method at `q = 0` and direct `q`-points computations
    for n_scell in [1, 2, 3]
        r_q0 = test_supercell_q0(; n_scell, max_radius)
        @assert length(r_q0) == n_scell
        ph_band_q0 = ph_bands[n_scell][n_points÷2+1]
        @test norm(r_q0 - ph_band_q0) < tolerance
    end
end
