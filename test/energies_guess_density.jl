# TODO Once we have converged SCF densities in a file it would be better to instead / also
#      test the energies of these densities and compare them directly to the reference
#      energies obtained in the data files

@testitem "Evaluate energies of guess density" setup=[TestCases] begin
    using DFTK
    silicon = TestCases.silicon

    Ecut = 15
    n_bands = 8
    fft_size = [27, 27, 27]
    kgrid  = (1, 2, 3)
    kshift = (0, 1/2, 0)

    model = model_DFT(silicon.lattice, silicon.atoms, silicon.positions,
                      [:lda_x, :lda_c_vwn]; symmetries=false)
    basis = PlaneWaveBasis(model; Ecut, kgrid, fft_size, kshift)

    ρ0 = guess_density(basis, ValenceDensityGaussian())
    E, H = energy_hamiltonian(basis, nothing, nothing; ρ=ρ0)

    @test E["Hartree"] ≈  0.3527293727197568  atol=5e-8
    @test E["Xc"]      ≈ -2.3033165870558165  atol=5e-8

    # Run one diagonalization and compute energies
    res = diagonalize_all_kblocks(lobpcg_hyper, H, n_bands, tol=1e-9)
    occupation = [[2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0]
                  for i = 1:length(basis.kpoints)]
    ρ = compute_density(H.basis, res.X, occupation)
    E, H = energy_hamiltonian(basis, res.X, occupation; ρ)

    @test E["Kinetic"]        ≈  3.3824289861522194  atol=5e-8
    @test E["AtomicLocal"]    ≈ -2.4178712046759157  atol=5e-8
    @test E["AtomicNonlocal"] ≈  1.664289455206788   atol=5e-8
    @test E["Hartree"]        ≈  0.6712993199211524  atol=5e-8
    @test E["Xc"]             ≈ -2.4489960475309056  atol=5e-8
    @test E["Ewald"]          ≈ -8.397893578467201   atol=5e-8
    @test E["PspCorrection"]  ≈ -0.294622067031369   atol=5e-8


    # Now we have a reasonable set of ψ, we make up a crazy model, and check the energies
    V(x, p) = 4*p.ε * ((p.σ/x)^12 - (p.σ/x)^6)
    params = Dict( (:Si, :Si) => (; ε=1e5, σ=0.5) )
    model = model_PBE(silicon.lattice, silicon.atoms, silicon.positions;
                      extra_terms=[ExternalFromReal(X -> cos(1.2 * (X[1] + X[3]))),
                                   ExternalFromFourier(X -> cos(1.3 * (X[1] + X[3]))),
                                   LocalNonlinearity(ρ -> 1.2 * ρ^2.4),
                                   Magnetic(X -> [1, cos(1.4 * X[2]), exp(X[3])]),
                                   PairwisePotential(V, params)],
                      )
    basis = PlaneWaveBasis(model; Ecut, kgrid, fft_size, kshift)
    E, H = energy_hamiltonian(basis, res.X, occupation; ρ)

    @test E["Kinetic"]             ≈  3.3824289861522194  atol=5e-8
    @test E["AtomicLocal"]         ≈ -2.4178712046759157  atol=5e-8
    @test E["AtomicNonlocal"]      ≈  1.664289455206788   atol=5e-8
    @test E["Hartree"]             ≈  0.6712993199211524  atol=5e-8
    @test E["Xc"]                  ≈ -2.469375219486637   atol=5e-8
    @test E["Ewald"]               ≈ -8.397893578467201   atol=5e-8
    @test E["PspCorrection"]       ≈ -0.294622067031369   atol=5e-8
    @test E["ExternalFromReal"]    ≈ -0.01756831422361496 atol=5e-8
    @test E["ExternalFromFourier"] ≈  0.06493077052321815 atol=5e-8
    @test E["LocalNonlinearity"]   ≈  0.14685350034704006 atol=5e-8
    @test E["PairwisePotential"]   ≈ -4.151269801749716   atol=5e-8

    # TODO This is not really a test ... and it does not really work stably.
    # @test E["Magnetic"]            ≈  1.99901120545585e-7 atol=5e-8
end
