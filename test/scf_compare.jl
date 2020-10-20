using Test
using DFTK

include("testcases.jl")

@testset "Compare different SCF algorithms (no spin, no temperature)" begin
    Ecut = 3
    n_bands = 6
    fft_size = [9, 9, 9]
    tol = 1e-7

    Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
    model = model_DFT(silicon.lattice, [Si => silicon.positions], [:lda_xc_teter93])
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)

    # Run nlsolve without guess
    ρ0 = RealFourierArray(basis)  # RFA of zeros
    ρ_nl = self_consistent_field(basis; ρ=ρ0, tol=tol).ρ.fourier

    # Run DM
    @testset "Direct minimization" begin
        ρ_dm = direct_minimization(basis; g_tol=tol).ρ.fourier
        @test maximum(abs.(ρ_dm - ρ_nl)) < sqrt(tol) / 10
    end

    # Run other SCFs with SAD guess
    ρ0 = guess_density(basis, [Si => silicon.positions])
    for solver in (scf_nlsolve_solver, scf_damping_solver, scf_anderson_solver,
                   scf_CROP_solver)
        @testset "Testing $solver" begin
            ρ_alg = self_consistent_field(basis; ρ=ρ0, solver=solver(), tol=tol).ρ.fourier
            @test maximum(abs.(ρ_alg - ρ_nl)) < sqrt(tol) / 10
        end
    end

    @testset "Enforce symmetry" begin
        ρ_alg = self_consistent_field(basis; ρ=ρ0, tol=tol, enforce_symmetry=true).ρ.fourier
        @test maximum(abs.(ρ_alg - ρ_nl)) < sqrt(tol) / 10
    end

    # Run other mixing with default solver (the others are too slow...)
    for mixing in (KerkerMixing(), SimpleMixing(), SimpleMixing(.5), DielectricMixing(εr=12),
                   HybridMixing(), HybridMixing(εr=10, RPA=false))
        @testset "Testing $mixing" begin
            ρ_alg = self_consistent_field(basis; ρ=ρ0, mixing=mixing, tol=tol).ρ.fourier
            @test maximum(abs.(ρ_alg - ρ_nl)) < sqrt(tol) / 10
        end
    end
end


@testset "Compare different SCF algorithms (collinear spin, temperature)" begin
    Ecut = 7
    n_bands = 8
    fft_size = [12, 12, 15]
    tol = 1e-7

    O = ElementPsp(o2molecule.atnum, psp=load_psp(o2molecule.psp))
    magnetic_moments = [O => [1., 1.]]
    model = model_LDA(o2molecule.lattice, [O => o2molecule.positions],
                      temperature=0.02, smearing=smearing=Smearing.Gaussian(),
                      magnetic_moments=magnetic_moments)
    basis = PlaneWaveBasis(model, Ecut; fft_size=fft_size, kgrid=[1, 1, 1])
    ρspin0 = guess_spin_density(basis, magnetic_moments)

    # Reference: Default algorithm
    ρ0     = guess_density(basis)
    scfres = self_consistent_field(basis, ρ=ρ0, ρspin=ρspin0, tol=tol)
    ρspin_ref = scfres.ρspin.fourier
    ρ_ref     = scfres.ρ.fourier

    for mixing in (KerkerMixing(), DielectricMixing(εr=10))
        @testset "Testing $mixing" begin
            scfres = self_consistent_field(basis; ρ=ρ0, ρspin=ρspin0, mixing=mixing, tol=tol)
            @test maximum(abs.(scfres.ρ.fourier     - ρ_ref    )) < sqrt(tol)
            @test maximum(abs.(scfres.ρspin.fourier - ρspin_ref)) < sqrt(tol)
        end
    end
end
