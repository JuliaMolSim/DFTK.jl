using Test
using DFTK

include("testcases.jl")

@testset "Compare different SCF algorithms" begin
    Ecut = 3
    n_bands = 6
    fft_size = [9, 9, 9]
    tol = 1e-7

    Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
    model = model_DFT(silicon.lattice, [Si => silicon.positions], [:lda_xc_teter93])
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)

    # Run nlsolve without guess
    ρ0 = RealFourierArray(basis)  # RFA of zeros
    scfres = self_consistent_field(basis; ρ=ρ0, tol=tol)
    ρ_nl = scfres.ρ.fourier

    # Run DM
    println("\nTesting direct minimization")
    dmres = direct_minimization(basis; g_tol=tol)
    ρ_dm = dmres.ρ.fourier
    @test maximum(abs.(ρ_dm - ρ_nl)) < sqrt(tol) / 10

    # Run other SCFs with SAD guess
    ρ0 = guess_density(basis, [Si => silicon.positions])
    for solver in (scf_nlsolve_solver, scf_damping_solver, scf_anderson_solver,
                   scf_CROP_solver)
        println("\nTesting $solver")
        scfres = self_consistent_field(basis; ρ=ρ0, solver=solver(), tol=tol)
        ρ_alg = scfres.ρ.fourier
        @test maximum(abs.(ρ_alg - ρ_nl)) < sqrt(tol) / 10
    end

    # Run other mixing with default solver (the others are too slow...)
    for mixing in (KerkerMixing(), SimpleMixing(), SimpleMixing(.5))
        println("\n Testing $mixing")
        scfres = self_consistent_field(basis; ρ=ρ0, mixing=mixing, tol=tol)
        ρ_alg = scfres.ρ.fourier
        @test maximum(abs.(ρ_alg - ρ_nl)) < sqrt(tol) / 10
    end
end
