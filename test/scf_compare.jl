using Test
using DFTK

include("testcases.jl")

@testset "Compare different SCF algorithms" begin
    Ecut = 3
    n_bands = 6
    fft_size = [9, 9, 9]
    tol = 1e-6

    Si = Element(silicon.atnum, psp=load_psp(silicon.psp))
    model = model_dft(silicon.lattice, :lda_xc_teter93, [Si => silicon.positions])
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)

    # Run nlsolve without guess
    scfres = self_consistent_field(Hamiltonian(basis), n_bands, tol=tol,
                                   solver=scf_nlsolve_solver())
    ρ_nl = scfres.ρ.fourier

    # Run DM
    println("\nTesting direct minimization")
    dmres = direct_minimization(basis; g_tol=1e-8)
    ρ_dm = dmres.ρ.fourier
    @test maximum(abs.(ρ_dm - ρ_nl)) < 30tol

    # Run other SCFs with SAD guess
    ρ0 = guess_density(basis, [Si => silicon.positions])
    for solver in (scf_nlsolve_solver, scf_damping_solver, scf_anderson_solver,
                   scf_CROP_solver)
        println("\nTesting $solver")
        scfres = self_consistent_field(Hamiltonian(basis, ρ0), n_bands, tol=tol, solver=solver())
        ρ_alg = scfres.ρ.fourier
        @test maximum(abs.(ρ_alg - ρ_nl)) < 30tol
    end

    # Run other mixing with nlsolve (the others are too slow...)
    for mixing in (KerkerMixing(), SimpleMixing(), SimpleMixing(.5))
        println("\n Testing $mixing")
        scfres = self_consistent_field(Hamiltonian(basis, ρ0), n_bands, tol=tol, solver=scf_nlsolve_solver(), mixing=mixing)
        ρ_alg = scfres.ρ.fourier
        @test maximum(abs.(ρ_alg - ρ_nl)) < 30tol
    end
end
