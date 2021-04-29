using Test
using DFTK
import DFTK: Applyχ0Model

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
    ρ0 = zeros(basis.fft_size..., 1)
    ρ_nl = self_consistent_field(basis; ρ=ρ0, tol=tol).ρ

    # Run DM
    if mpi_nprocs() == 1  # Distributed implementation not yet available
        @testset "Direct minimization" begin
            ρ_dm = direct_minimization(basis; g_tol=tol).ρ
            @test maximum(abs.(ρ_dm - ρ_nl)) < sqrt(tol) / 10
        end
    end

    # Run other SCFs with SAD guess
    ρ0 = guess_density(basis, [Si => silicon.positions])
    for solver in (scf_nlsolve_solver(), scf_damping_solver(1.2), scf_anderson_solver(),
                   scf_CROP_solver())
        @testset "Testing $solver" begin
            ρ_alg = self_consistent_field(basis; ρ=ρ0, solver=solver, tol=tol).ρ
            @test maximum(abs.(ρ_alg - ρ_nl)) < sqrt(tol) / 10
        end
    end

    @testset "Enforce symmetry" begin
        ρ_alg = self_consistent_field(basis; ρ=ρ0, tol=tol, enforce_symmetry=true).ρ
        @test maximum(abs.(ρ_alg - ρ_nl)) < sqrt(tol) / 10
    end

    # Run other mixing with default solver (the others are too slow...)
    for mixing in (KerkerMixing(), SimpleMixing(), SimpleMixing(.5), DielectricMixing(εr=12),
                   KerkerDosMixing(), HybridMixing(), HybridMixing(εr=10, RPA=false),
                   χ0Mixing(χ0terms=[Applyχ0Model()], RPA=true))
        @testset "Testing $mixing" begin
            ρ_alg = self_consistent_field(basis; ρ=ρ0, mixing=mixing, tol=tol).ρ
            @test maximum(abs.(ρ_alg - ρ_nl)) < sqrt(tol) / 10
        end
    end
end


@testset "Compare different SCF algorithms (no spin, temperature)" begin
    Ecut = 3
    n_bands = 6
    fft_size = [9, 9, 9]
    tol = 1e-7

    Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
    model = model_DFT(silicon.lattice, [Si => silicon.positions], [:lda_xc_teter93],
                      temperature=0.01, smearing=Smearing.Gaussian())
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)

    # Reference: Default algorithm
    ρ0    = guess_density(basis)
    ρ_ref = self_consistent_field(basis, ρ=ρ0, tol=tol).ρ

    for mixing in (KerkerDosMixing(), HybridMixing(RPA=true), HybridMixing(εr=10, RPA=false), )
        @testset "Testing $mixing" begin
            ρ_mix = self_consistent_field(basis; ρ=ρ0, mixing=mixing, tol=tol).ρ
            @test maximum(abs.(ρ_mix - ρ_ref)) < sqrt(tol)
        end
    end
end


@testset "Compare different SCF algorithms (collinear spin, temperature)" begin
    Ecut = 11
    n_bands = 8
    fft_size = [13, 13, 13]
    tol = 1e-7

    Fe = ElementPsp(iron_bcc.atnum, psp=load_psp(iron_bcc.psp))
    magnetic_moments = [Fe => [4.0]]
    model = model_LDA(iron_bcc.lattice, [Fe => iron_bcc.positions],
                      temperature=0.01, magnetic_moments=magnetic_moments,
                      spin_polarization=:collinear)
    basis = PlaneWaveBasis(model, Ecut; fft_size=fft_size, kgrid=[3, 3, 3])

    # Reference: Default algorithm
    ρ0     = guess_density(basis)
    scfres = self_consistent_field(basis, ρ=ρ0, tol=tol)
    ρ_ref = scfres.ρ

    for mixing in (KerkerMixing(), KerkerDosMixing(α=1.0), DielectricMixing(εr=10),
                   HybridMixing(εr=10), χ0Mixing(χ0terms=[Applyχ0Model()], RPA=false),)
        @testset "Testing $mixing" begin
            scfres = self_consistent_field(basis; ρ=ρ0, mixing=mixing, tol=tol)
            @test maximum(abs.(scfres.ρ - ρ_ref)) < sqrt(tol)
        end
    end
end
