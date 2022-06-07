using Test
using DFTK
using LinearAlgebra
using ForwardDiff
using StaticArrays

# ## Helper functions
# Some functions that will be helpful for this example.
fold(x)   = hcat(x...)
unfold(x) = Vec3.(eachcol(x))

const MAX_RADIUS = 1e3
const TOLERANCE = 1e-6
const N_POINTS = 10

function prepare_system(; case=1)
    positions = [[0.,0,0]]
    for i in 1:case-1
        push!(positions, i*ones(3)/case)
    end

    a = 5. * length(positions)
    lattice = a * [[1 0 0.]; [0 0 0.]; [0 0 0.]]

    s = DFTK.compute_inverse_lattice(lattice)
    if case === 1
        directions = [[s * [1,0,0]]]
    elseif case === 2
        directions = [[s * [1,0,0], s * [0,0,0]],
                      [s * [0,0,0], s * [1,0,0]]]
    elseif case === 3
        directions = [[s * [1,0,0], s * [0,0,0], s * [0,0,0]],
                      [s * [0,0,0], s * [1,0,0], s * [0,0,0]],
                      [s * [0,0,0], s * [0,0,0], s * [1,0,0]]]
    end

    params = Dict((:X, :X) => (; ε=1, σ=a / length(positions) /2^(1/6)))
    V(x, p) = 4*p.ε * ((p.σ/x)^12 - (p.σ/x)^6)

    (positions=positions, lattice=lattice, directions=directions, params=params, V=V)
end

# Compute phonons for a one-dimensional pairwise potential for a set of `q = 0` using
# supercell method
function test_supercell_q0(; N_scell=1)
    blob = prepare_system(; case=N_scell)
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
            n_positions = unfold(fold(positions) .+ ε .* s * direction)
            forces = zeros(Vec3{complex(eltype(ε))}, length(positions))
            DFTK.energy_pairwise(lattice, [:X for _ in positions],
                                 n_positions, V, params; forces=forces, max_radius=MAX_RADIUS)
            [(s * f)[1] for f in forces]
        end
    end
    sqrt.(abs.(eigvals(Φ)))
end

# Compute phonons for a one-dimensional pairwise potential for a set of `q`-points
function test_ph_disp(; case=1)
    blob = prepare_system(; case=case)
    positions = blob.positions
    lattice = blob.lattice
    directions = blob.directions
    params = blob.params
    V = blob.V

    pairwise_ph = (q, d; forces=nothing) ->
                     DFTK.energy_pairwise(lattice, [:X for _ in positions],
                                          positions, V, params; q=[q, 0, 0],
                                          ph_disp=d, forces=forces,
                                          max_radius=MAX_RADIUS)

    ph_bands = []
    qs = -1/2:1/N_POINTS:1/2
    for q in qs
        as = ComplexF64[]
        for d in directions
            res = -ForwardDiff.derivative(0.0) do ε
                forces = zeros(Vec3{complex(eltype(ε))}, length(positions))
                pairwise_ph(q, ε*d; forces=forces)
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
    ph_bands_1 = test_ph_disp(; case=1)
    ph_bands_2 = test_ph_disp(; case=2)
    ph_bands_3 = test_ph_disp(; case=3)

    min_1 = minimum(hcat(ph_bands_1...))
    max_1 = maximum(hcat(ph_bands_1...))
    min_2 = minimum(hcat(ph_bands_2...))
    max_2 = maximum(hcat(ph_bands_2...))
    min_3 = minimum(hcat(ph_bands_3...))
    max_3 = maximum(hcat(ph_bands_3...))

    # Recover the same extremum for the system whatever case we test
    @test ≈(min_1, min_2, atol=TOLERANCE)
    @test ≈(min_1, min_3, atol=TOLERANCE)
    @test ≈(max_1, max_2, atol=TOLERANCE)
    @test ≈(max_1, max_3, atol=TOLERANCE)

    r1_q0 = test_supercell_q0(; N_scell=1)
    @assert length(r1_q0) == 1
    ph_bands_1_q0 = ph_bands_1[N_POINTS÷2+1]
    @test norm(r1_q0 - ph_bands_1_q0) < TOLERANCE

    r2_q0 = sort(test_supercell_q0(; N_scell=2))
    @assert length(r2_q0) == 2
    ph_bands_2_q0 = ph_bands_2[N_POINTS÷2+1]
    @test norm(r2_q0 - ph_bands_2_q0) < TOLERANCE

    r3_q0 = sort(test_supercell_q0(; N_scell=3))
    @assert length(r3_q0) == 3
    ph_bands_3_q0 = ph_bands_3[N_POINTS÷2+1]
    @test norm(r3_q0 - ph_bands_3_q0) < TOLERANCE
end
