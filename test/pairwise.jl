using Test
using DFTK
using LinearAlgebra
using Random

@testset "Pairwise forces" begin
    a = 5.131570667152971
    lattice = a .* [0 1 1; 1 0 1; 1 1 0]
    # perturb positions away from equilibrium to get nonzero force
    positions = [ones(3)/8+rand(3)/20, -ones(3)/8]
    atom_types = [:Li, :H]
    V(x, p) = 4*p.ε * ((p.σ/x)^12 - (p.σ/x)^6)
    params = Dict((:Li, :H ) => (; ε=1, σ=2),
                  ( :H, :H ) => (; ε=1, σ=2),
                  (:Li, :Li) => (; ε=1, σ=2))
    term = PairwisePotential(V, params)

    # Test that the constructor has ordered the tuples
    @test (:H, :Li) ∈ keys(term.params)

    atoms = [ElementCoulomb(:Li) => [positions[1]],
             ElementCoulomb(:H)  => [positions[2]]]
    model = Model(lattice; atoms=atoms, terms=[term])
    basis = PlaneWaveBasis(model; Ecut=20, kgrid=(1, 1, 1))

    forces = hcat(compute_forces(basis.terms[1], basis, nothing, nothing)...)

    γ1 = DFTK.energy_pairwise(lattice, atom_types, positions,
                              term.V, term.params, term.max_radius)

    # Compare forces to finite differences
    disp = [rand(3)/20, rand(3)/20]
    ε = 1e-8
    γ2 = DFTK.energy_pairwise(lattice, atom_types, positions .+ ε .* disp,
                              term.V, term.params, term.max_radius)
    @test (γ2-γ1)/ε ≈ -dot(disp, forces) atol=abs(γ1*1e-6)
end
