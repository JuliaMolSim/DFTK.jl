@testitem "Anyons: check magnetic potential" begin
    using DFTK
    using LinearAlgebra

    # Test that the magnetic potential satisfies ∇∧A = 2π ρref, ∇⋅A = 0
    x = 1.23
    y = -1.8
    ε = 1e-8
    M = 2.31
    σ = 1.81
    dAdx = (  DFTK.magnetic_potential_produced_by_ρref(x + ε, y, M, σ)
            - DFTK.magnetic_potential_produced_by_ρref(x,     y, M, σ)) / ε
    dAdy = (  DFTK.magnetic_potential_produced_by_ρref(x, y + ε, M, σ)
            - DFTK.magnetic_potential_produced_by_ρref(x, y,     M, σ)) / ε
    curlA = dAdx[2] - dAdy[1]
    divA = dAdx[1] + dAdy[2]
    @test norm(curlA - 2π*DFTK.ρref_real_2D(x, y, M, σ)) < 1e-4
    @test abs(divA) < 1e-6
end

# Direct minimisation not supported on mpi
@testitem "Anyons: check E11" tags=[:dont_test_mpi] begin
    using DFTK
    using StaticArrays

    # See https://arxiv.org/pdf/1901.10739.pdf
    # We test E11, which is a quantity defined in the above paper

    ## Unit cell. Having one of the lattice vectors as zero means a 2D system
    a = 14
    lattice = a .* [[1 0 0.]; [0 1 0]; [0 0 0]];

    ## Confining scalar potential
    pot(x, y, z) = ((x - a/2)^2 + (y - a/2)^2);

    ## Parameters
    Ecut = 30
    n_electrons = 1
    β = 5;

    ## Collect all the terms, build and run the model
    terms = [Kinetic(; scaling_factor=2),
             ExternalFromReal(X -> pot(X...)),
             Anyonic(1, β)
             ]
    model = Model(lattice; n_electrons, terms, spin_polarization=:spinless)  # "spinless electrons"
    basis = PlaneWaveBasis(model; Ecut, kgrid=(1, 1, 1))
    scfres = direct_minimization(basis; tol=1e-6, maxiter=300)  # Limit maxiter as guess can be bad
    E = scfres.energies.total
    s = 2
    E11 = π/2 * (2(s+1)/s)^((s+2)/s) * (s/(s+2))^(2(s+1)/s) * E^((s+2)/s) / β
    @test 1.1 ≤ E11/(2π) ≤ 1.3 # 1.18 in the paper
end
