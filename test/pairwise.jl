using Test
using DFTK
using LinearAlgebra
using Random
Random.seed!(0)

@testset "Forces" begin
    lattice = [0.0  5.131570667152971 5.131570667152971;
               5.131570667152971 0.0 5.131570667152971;
               5.131570667152971 5.131570667152971  0.0]
    # perturb positions away from equilibrium to get nonzero force
    positions = [ones(3)/8+rand(3)/20, -ones(3)/8]
    charges = [14, 14]
    atom_types = map(ElementCoulomb, charges)
    V(x, p) = 4*p.ε * ((p.σ/x)^12 - (p.σ/x)^6)
    params = Dict((atom_types[1], atom_types[1]) => (; ε=1, σ=2))
    term = DFTK.PairwisePotential(V, params)

    forces = zeros(Vec3{Float64}, 2)
    γ1 = DFTK.energy_pairwise(lattice, atom_types, positions, term.V, term.params, term.max_radius; forces=forces)

    # Compare forces to finite differences
    disp = [rand(3)/20, rand(3)/20]
    ε = 1e-8
    γ2 = DFTK.energy_pairwise(lattice, atom_types, positions .+ ε .* disp, term.V, term.params, term.max_radius)
    @test (γ2-γ1)/ε ≈ -dot(disp, forces) atol=abs(γ1*1e-6)
end

@testset "PairwisePotential integration test" begin
    a = 10.0
    lattice = a .* [[1 0 0.]; [0 1 0]; [0 0 1]]
    nucleus = ElementCoulomb(:Si)
    atoms = [nucleus => [[0., 0, 0], ones(3)/2]]

    V(x, p) = 4*p.ε * ((p.σ/x)^12 - (p.σ/x)^6)
    params = Dict((nucleus, nucleus) => (; ε=1e5, σ=0.5))

    terms = [
             DFTK.PairwisePotential(V, params),
            ]
    model = model_atomic(lattice, atoms; extra_terms=terms, spin_polarization=:spinless,
                         symmetries=false)

    Ecut = 30
    basis = PlaneWaveBasis(model; Ecut, kgrid=(1, 1, 1))
    scfres = self_consistent_field(basis, tol=1e-6)
    @test scfres.energies.total ≈ - 319.7809362482486 atol=1e-6
    forces = DFTK.compute_forces(scfres)
    @test norm(forces) ≈ 0.000541366326365431 atol=1e-6
end
