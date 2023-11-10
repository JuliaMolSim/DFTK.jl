@testitem "Pairwise forces" begin
    using DFTK
    using DFTK: energy_forces_pairwise
    using LinearAlgebra

    a = 5.131570667152971
    lattice = a .* [0 1 1; 1 0 1; 1 1 0]
    # perturb positions away from equilibrium to get nonzero force
    atoms     = [ElementCoulomb(:Li), ElementCoulomb(:H)]
    positions = [ones(3)/8+rand(3)/20, -ones(3)/8]
    symbols   = [:Li, :H]
    V(x, p) = 4*p.ε * ((p.σ/x)^12 - (p.σ/x)^6)
    params = Dict((:Li, :H ) => (; ε=1, σ=2),
                  ( :H, :H ) => (; ε=1, σ=2),
                  (:Li, :Li) => (; ε=1, σ=2))
    term = PairwisePotential(V, params)

    # Test that the constructor has ordered the tuples
    @test (:H, :Li) ∈ keys(term.params)

    model = Model(lattice, atoms, positions; terms=[term])
    basis = PlaneWaveBasis(model; Ecut=20, kgrid=(1, 1, 1))
    forces = compute_forces(only(basis.terms), basis, nothing, nothing)

    # Compare forces to finite differences
    ε=1e-8
    disp=[rand(3) / 20, rand(3) / 20]
    T = eltype(lattice)
    q = zeros(3)
    E1 = energy_forces_pairwise(T, lattice, symbols, positions,              term.V,
                                term.params, q, nothing).energy
    E2 = energy_forces_pairwise(T, lattice, symbols, positions .+ ε .* disp, term.V,
                                term.params, q, nothing).energy
    @test (E2 - E1) / ε ≈ -dot(disp, forces) atol=abs(1e-6E1)
end
