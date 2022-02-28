using Test
using DFTK
using LinearAlgebra
using Random

@testset "Hydrogen atom" begin
    lattice = 16 * Diagonal(ones(3))
    positions = [[0,0,0]]
    charges = [1]
    atom_types = map(ElementCoulomb, charges)
    V(x, p) = 4*p.ε * ((p.σ/x)^12 - (p.σ/x)^6)
    params = Dict((atom_types[1], atom_types[1]) => (; ε=1, σ=16))
    term = DFTK.PairwisePotential(V, params)

    ref = -4.365199475203764
    γ_E = DFTK.energy_pairwise(lattice, atom_types, positions, term.V, term.params, term.max_radius)
    @test abs(γ_E - ref) < 1e-8
end

@testset "Hydrogen atom 1D" begin
    lattice = Diagonal(zeros(3))
    non_zero_dim = Random.rand([1:3]...)
    lattice[non_zero_dim, non_zero_dim] = 16
    positions = [[0,0,0]]
    charges = [1]
    atom_types = map(ElementCoulomb, charges)
    V(x, p) = 4*p.ε * ((p.σ/x)^12 - (p.σ/x)^6)
    params = Dict((atom_types[1], atom_types[1]) => (; ε=1, σ=16))
    term = DFTK.PairwisePotential(V, params)

    ref = -0.06832091898611523
    γ_E = DFTK.energy_pairwise(lattice, atom_types, positions, term.V, term.params, term.max_radius)
    @test abs(γ_E - ref) < 1e-8
end

@testset "Hydrogen atom 2D" begin
    lattice = 16 * Diagonal(ones(3))
    zero_dim = Random.rand([1:3]...)
    lattice[zero_dim, zero_dim] = 0
    positions = [[0,0,0]]
    charges = [1]
    atom_types = map(ElementCoulomb, charges)
    V(x, p) = 4*p.ε * ((p.σ/x)^12 - (p.σ/x)^6)
    params = Dict((atom_types[1], atom_types[1]) => (; ε=1, σ=16))
    term = DFTK.PairwisePotential(V, params)

    ref = -1.1876602197755037
    γ_E = DFTK.energy_pairwise(lattice, atom_types, positions, term.V, term.params, term.max_radius)
    @test abs(γ_E - ref) < 1e-8
end

@testset "Silicon diamond structure" begin
    lattice = [0.0  5.131570667152971 5.131570667152971;
               5.131570667152971 0.0 5.131570667152971;
               5.131570667152971 5.131570667152971  0.0]
    positions = [ones(3)/8, -ones(3)/8]
    charges = [14, 14]
    atom_types = map(ElementCoulomb, charges)
    V(x, p) = 4*p.ε * ((p.σ/x)^12 - (p.σ/x)^6)
    params = Dict((atom_types[1], atom_types[1]) => (; ε=1, σ=2))
    term = DFTK.PairwisePotential(V, params)

    ref = -0.1689166856766943
    γ_E = DFTK.energy_pairwise(lattice, atom_types, positions, term.V, term.params, term.max_radius)
    @test abs(γ_E - ref) < 1e-8
end

@testset "Boron molecule" begin
    lattice = 16 * Diagonal(ones(3))
    positions = [[0,0,0], [0.14763485355139283, 0, 0]]
    charges = [5, 5]
    atom_types = map(ElementCoulomb, charges)
    V(x, p) = 4*p.ε * ((p.σ/x)^12 - (p.σ/x)^6)
    params = Dict((atom_types[1], atom_types[1]) => (; ε=1, σ=2))
    term = DFTK.PairwisePotential(V, params)

    ref = -0.9310016711961596
    γ_E = DFTK.energy_pairwise(lattice, atom_types, positions, term.V, term.params, term.max_radius)
    @test abs(γ_E - ref) < 1e-7
end

@testset "Hydrogen molecule" begin
    lattice = 16 * Diagonal(ones(3))
    positions = [
        [0.45312500031210007, 1/2, 1/2],
        [0.5468749996028622, 1/2, 1/2],
    ]
    charges = [1, 1]
    atom_types = map(ElementCoulomb, charges)
    V(x, p) = 4*p.ε * ((p.σ/x)^12 - (p.σ/x)^6)
    params = Dict((atom_types[1], atom_types[1]) => (; ε=1, σ=1))
    term = DFTK.PairwisePotential(V, params)

    ref = -0.3203406836256344
    γ_E = DFTK.energy_pairwise(lattice, atom_types, positions, term.V, term.params, term.max_radius)
    @test abs(γ_E - ref) < 1e-7
end

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
    @test abs(scfres.energies.total + 319.7809362482486) < 1e-6
    forces = DFTK.compute_forces_cart(scfres)
    @test abs(norm(forces) - 4.014438647235789e-5) < 1e-8
end
