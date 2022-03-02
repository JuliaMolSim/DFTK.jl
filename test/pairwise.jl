using Test
using DFTK
using LinearAlgebra
using Random

@testset "Forces" begin
    lattice = [0.0  5.131570667152971 5.131570667152971;
               5.131570667152971 0.0 5.131570667152971;
               5.131570667152971 5.131570667152971  0.0]
    # perturb positions away from equilibrium to get nonzero force
    positions = [ones(3)/8+rand(3)/20, -ones(3)/8]
    charges = [14, 14]
    atom_types = map(x -> ElementCoulomb(x).symbol, charges)
    V(x, p) = 4*p.ε * ((p.σ/x)^12 - (p.σ/x)^6)
    params = Dict((atom_types[1], atom_types[1]) => (; ε=1, σ=2))
    term = PairwisePotential(V, params)

    # We only provide the necessary information.
    atoms = [ElementCoulomb(14) => positions]
    model = Model(lattice; atoms=atoms, terms=[term])
    basis = PlaneWaveBasis(model; Ecut=20, kgrid=(1, 1, 1))

    forces = hcat(compute_forces(basis.terms[1], basis, nothing, nothing)...)

    γ1 = DFTK.energy_pairwise(lattice, atom_types, positions, term.V, term.params, term.max_radius)

    # Compare forces to finite differences
    disp = [rand(3)/20, rand(3)/20]
    ε = 1e-8
    γ2 = DFTK.energy_pairwise(lattice, atom_types, positions .+ ε .* disp, term.V, term.params, term.max_radius)
    @test (γ2-γ1)/ε ≈ -dot(disp, forces) atol=abs(γ1*1e-6)
end
