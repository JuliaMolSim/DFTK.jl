using Test
using DFTK
import DFTK: Applyχ0Model, select_occupied_orbitals

include("testcases.jl")

@testset "Compare different SCF algorithms (no spin, no temperature)" begin
    Ecut = 3
    n_bands = 6
    fft_size = [9, 9, 9]
    tol = 1e-7

    model = model_LDA(silicon.lattice, silicon.atoms, silicon.positions)
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.kweights; fft_size)

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

    # Run Newton algorithm
    if mpi_nprocs() == 1  # Distributed implementation not yet available
        @testset "Newton" begin
            scfres_start = self_consistent_field(basis, maxiter=1)
            # remove virtual orbitals
            ψ0, _ = select_occupied_orbitals(basis, scfres_start.ψ,
                                             scfres_start.occupation)
            ρ_newton = newton(basis, ψ0; tol=tol).ρ
            @test maximum(abs.(ρ_newton - ρ_nl)) < sqrt(tol) / 10
        end
    end

    # Run other SCFs with SAD guess
    ρ0 = guess_density(basis)
    for solver in (scf_nlsolve_solver(), scf_damping_solver(1.2), scf_anderson_solver(),
                   scf_CROP_solver())
        @testset "Testing $solver" begin
            ρ_alg = self_consistent_field(basis; ρ=ρ0, solver=solver, tol=tol).ρ
            @test maximum(abs.(ρ_alg - ρ_nl)) < sqrt(tol) / 10
        end
    end

    # Run other mixing with default solver (the others are too slow...)
    for mixing in (KerkerMixing(), SimpleMixing(), DielectricMixing(εr=12),
                   KerkerDosMixing(), HybridMixing(), HybridMixing(εr=10, RPA=false),
                   χ0Mixing(χ0terms=[Applyχ0Model()], RPA=true))
        @testset "Testing $mixing" begin
            ρ_alg = self_consistent_field(basis; ρ=ρ0, mixing=mixing, tol=tol, damping=0.8).ρ
            @test maximum(abs.(ρ_alg - ρ_nl)) < sqrt(tol) / 10
        end
    end

    # Potential mixing
    scfres = DFTK.scf_potential_mixing(basis, mixing=KerkerMixing(), tol=tol, ρ=ρ0)
    @test maximum(abs.(scfres.ρ - ρ_nl)) < sqrt(tol) / 10

    # Adaptive potential mixing
    scfres = DFTK.scf_potential_mixing_adaptive(basis, mixing=SimpleMixing(), tol=tol, ρ=ρ0)
    @test maximum(abs.(scfres.ρ - ρ_nl)) < sqrt(tol) / 10
end

@testset "Compare different SCF algorithms (collinear spin, no temperature)" begin
    Ecut = 3
    n_bands = 6
    fft_size = [9, 9, 9]
    tol = 1e-7

    magnetic_moments = [1, 1]
    model = model_LDA(silicon.lattice, silicon.atoms, silicon.positions; magnetic_moments)
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.kweights; fft_size)
    ρ_nl = self_consistent_field(basis; tol=tol).ρ

    # Run DM
    if mpi_nprocs() == 1  # Distributed implementation not yet available
        @testset "Direct minimization" begin
            ρ_dm = direct_minimization(basis; g_tol=tol).ρ
            @test maximum(abs.(ρ_dm - ρ_nl)) < sqrt(tol) / 10
        end
    end

    # Run Newton algorithm
    if mpi_nprocs() == 1  # Distributed implementation not yet available
        @testset "Newton" begin
            scfres_start = self_consistent_field(basis, maxiter=1)
            # remove virtual orbitals
            ψ0, _ = select_occupied_orbitals(basis, scfres_start.ψ,
                                             scfres_start.occupation)
            ρ_newton = newton(basis, ψ0; tol=tol).ρ
            @test maximum(abs.(ρ_newton - ρ_nl)) < sqrt(tol) / 10
        end
    end
end

@testset "Compare different SCF algorithms (no spin, temperature)" begin
    Ecut = 3
    n_bands = 6
    fft_size = [9, 9, 9]
    tol = 1e-7

    model = model_LDA(silicon.lattice, silicon.atoms, silicon.positions;
                      temperature=0.01, smearing=Smearing.Gaussian())
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.kweights; fft_size)

    # Reference: Default algorithm
    ρ0    = guess_density(basis)
    ρ_ref = self_consistent_field(basis, ρ=ρ0, tol=tol).ρ

    for mixing in (KerkerDosMixing(), HybridMixing(RPA=true), LdosMixing(RPA=false),
                   HybridMixing(εr=10, RPA=true), )
        @testset "Testing $mixing" begin
            ρ_mix = self_consistent_field(basis; ρ=ρ0, mixing=mixing, tol=tol, damping=0.8).ρ
            @test maximum(abs.(ρ_mix - ρ_ref)) < sqrt(tol)
        end
    end
end


@testset "Compare different SCF algorithms (collinear spin, temperature)" begin
    n_bands = 8
    fft_size = [13, 13, 13]
    tol = 1e-7

    magnetic_moments = [4.0]
    model = model_LDA(iron_bcc.lattice, iron_bcc.atoms, iron_bcc.positions;
                      temperature=0.01, magnetic_moments, spin_polarization=:collinear)
    basis = PlaneWaveBasis(model; Ecut=11, fft_size=fft_size, kgrid=[3, 3, 3])

    # Reference: Default algorithm
    ρ0     = guess_density(basis, magnetic_moments)
    scfres = self_consistent_field(basis, ρ=ρ0, tol=tol)
    ρ_ref  = scfres.ρ

    for mixing in (KerkerMixing(), KerkerDosMixing(), DielectricMixing(εr=10),
                   HybridMixing(εr=10), χ0Mixing(χ0terms=[Applyχ0Model()], RPA=false),)
        @testset "Testing $mixing" begin
            scfres = self_consistent_field(basis; ρ=ρ0, mixing=mixing, tol=tol, damping=0.8)
            @test maximum(abs.(scfres.ρ - ρ_ref)) < 2sqrt(tol)
        end
    end

    # Potential mixing
    scfres = DFTK.scf_potential_mixing(basis, mixing=KerkerMixing(), tol=tol, ρ=ρ0)
    @test maximum(abs.(scfres.ρ - ρ_ref)) < sqrt(tol)

    # Adaptive potential mixing (started deliberately with the very bad damping
    #          of 1.5 to provoke backtrack steps ... don't do this in production runs!)
    scfres = DFTK.scf_potential_mixing_adaptive(basis, mixing=SimpleMixing(), tol=tol, ρ=ρ0,
                                                damping=DFTK.AdaptiveDamping(1.5))
    @test maximum(abs.(scfres.ρ - ρ_ref)) < sqrt(tol)
end
