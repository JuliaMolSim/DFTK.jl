using Test
using DFTK

include("testcases.jl")

# TODO Once we have converged SCF densities in a file it would be better to instead / also
#      test the energies of these densities and compare them directly to the reference
#      energies obtained in the data files

@testset "Evaluate energies of guess density" begin
    Ecut = 15
    n_bands = 8
    fft_size = [27, 27, 27]

    Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
    model = model_DFT(silicon.lattice, [Si => silicon.positions], [:lda_x, :lda_c_vwn])
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)

    ρ0 = guess_density(basis, [Si => silicon.positions])
    E, H = energy_hamiltonian(basis, nothing, nothing; ρ=ρ0)

    @test E["Hartree"] ≈  0.3527293727197568  atol=5e-8
    @test E["Xc"]      ≈ -2.3033165870558165  atol=5e-8

    # Run one diagonalization and compute energies
    res = diagonalize_all_kblocks(lobpcg_hyper, H, n_bands, tol=1e-9)
    occupation = [[2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0]
                  for i in 1:length(basis.kpoints)]
    ρnew = compute_density(H.basis, res.X, occupation)
    E, H = energy_hamiltonian(basis, res.X, occupation; ρ=ρnew)

    @test E["Kinetic"]        ≈  3.291847293270256   atol=5e-8
    @test E["AtomicLocal"]    ≈ -2.367978663117999   atol=5e-8
    @test E["AtomicNonlocal"] ≈  1.6527493682542034  atol=5e-8
    @test E["Hartree"]        ≈  0.6477025793366571  atol=5e-8
    @test E["Xc"]             ≈ -2.4375329416161162  atol=5e-8
    @test E["Ewald"]          ≈ -8.397893578467201   atol=5e-8
    @test E["PspCorrection"]  ≈ -0.294622067031369   atol=5e-8


    # Now we have a reasonable set of ψ, we make up a crazy model, and check the energies
    model = model_DFT(silicon.lattice,
                      [Si => silicon.positions],
                      [:gga_x_pbe, :gga_c_pbe],
                      extra_terms=[ExternalFromReal(X -> cos(1.2*(X[1]+X[3]))),
                                   ExternalFromFourier(X -> cos(1.3*(X[1]+X[3]))),
                                   LocalNonlinearity(ρ -> 1.2 * ρ^2.4),
                                   Magnetic(X -> [1, cos(1.4*X[2]), exp(X[3])])]
                      )
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)

    E, H = energy_hamiltonian(basis, res.X, occupation; ρ=ρnew)

    @test E["Kinetic"]             ≈  3.291847293270256   atol=5e-8
    @test E["AtomicLocal"]         ≈ -2.367978663117999   atol=5e-8
    @test E["AtomicNonlocal"]      ≈  1.6527493682542034  atol=5e-8
    @test E["Hartree"]             ≈  0.6477025793366571  atol=5e-8
    @test E["Xc"]                  ≈ -2.456212919662419   atol=5e-8
    @test E["Ewald"]               ≈ -8.397893578467201   atol=5e-8
    @test E["PspCorrection"]       ≈ -0.294622067031369   atol=5e-8
    @test E["ExternalFromReal"]    ≈  0.139216686139006   atol=5e-8
    @test E["ExternalFromFourier"] ≈  0.057896835498415   atol=5e-8
    @test E["LocalNonlinearity"]   ≈  0.142649748399169   atol=5e-8
    @test E["Magnetic"]            ≈ -451.5652707506372   atol=5e-7
end
