using Test
using DFTK
import DFTK: solve_ΩplusK, apply_Ω, apply_K, solve_ΩplusK_split
import DFTK: filled_occupation, compute_projected_gradient, compute_occupation
import DFTK: select_occupied_orbitals

include("testcases.jl")

@testset "ΩplusK" begin
    Ecut = 3
    fft_size = [9, 9, 9]
    model = model_DFT(silicon.lattice, silicon.atoms, silicon.positions, [:lda_xc_teter93])
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.kweights; fft_size)
    scfres = self_consistent_field(basis; tol=10)

    ψ, occupation = select_occupied_orbitals(basis, scfres.ψ, scfres.occupation)

    ρ = compute_density(basis, ψ, occupation)
    rhs = compute_projected_gradient(basis, ψ, occupation)
    ϕ = rhs + ψ

    @testset "self-adjointness of solve_ΩplusK" begin
        @test isapprox(
            real(dot(ϕ, solve_ΩplusK(basis, ψ, rhs, occupation).δψ)),
            real(dot(solve_ΩplusK(basis, ψ, ϕ, occupation).δψ, rhs)),
            atol=1e-7
        )
    end

    @testset "self-adjointness of apply_Ω" begin
        _, H = energy_hamiltonian(basis, ψ, occupation; ρ=ρ)
        # Rayleigh-coefficients
        Λ = [ψk'Hψk for (ψk, Hψk) in zip(ψ, H * ψ)]

        # Ω is complex-linear and so self-adjoint as a complex operator.
        @test isapprox(
            dot(ϕ, apply_Ω(rhs, ψ, H, Λ)),
            dot(apply_Ω(ϕ, ψ, H, Λ), rhs),
            atol=1e-7
        )
    end

    @testset "self-adjointness of apply_K" begin
        # K involves conjugates and is only a real-linear operator,
        # hence we test using the real dot product.
        @test isapprox(
            real(dot(ϕ, apply_K(basis, rhs, ψ, ρ, occupation))),
            real(dot(apply_K(basis, ϕ, ψ, ρ, occupation), rhs)),
            atol=1e-7
        )
    end

    @testset "ΩplusK_split, 0K" begin
        Ecut = 3
        fft_size = [9, 9, 9]
        model = model_DFT(silicon.lattice, silicon.atoms, silicon.positions, [:lda_xc_teter93])
        basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.kweights; fft_size)
        scfres = self_consistent_field(basis; tol=10)

        ψ = scfres.ψ
        rhs = compute_projected_gradient(basis, scfres.ψ, scfres.occupation)
        ϕ = rhs + ψ

        @testset "self-adjointness of solve_ΩplusK_split" begin
            @test isapprox(real(dot(ϕ, solve_ΩplusK_split(scfres, rhs).δψ)),
                           real(dot(solve_ΩplusK_split(scfres, ϕ).δψ, rhs)),
                           atol=1e-7)
        end

        @testset "solve_ΩplusK_split <=> solve_ΩplusK" begin
            scfres = self_consistent_field(basis; tol=1e-10)
            ψ, occupation = select_occupied_orbitals(basis, scfres.ψ, scfres.occupation)
            rhs, _ = select_occupied_orbitals(basis, rhs, occupation)
            δψ1 = solve_ΩplusK(basis, ψ, rhs, occupation).δψ
            δψ2 = solve_ΩplusK_split(scfres, rhs).δψ
            δψ2, _ = select_occupied_orbitals(basis, δψ2, occupation)
            @test norm(δψ1 - δψ2) < 1e-7
        end

    end

    @testset "ΩplusK_split, temp" begin
        Ecut = 5
        fft_size = [9, 9, 9]
        model = model_DFT(magnesium.lattice, magnesium.atoms, magnesium.positions,
                          [:lda_xc_teter93]; temperature=magnesium.temperature)
        basis = PlaneWaveBasis(model, Ecut, magnesium.kcoords, magnesium.kweights; fft_size)
        scfres = self_consistent_field(basis; n_bands=7, tol=1e-12,
                                       occupation_threshold=1e-10)

        ψ = scfres.ψ
        rhs = compute_projected_gradient(basis, scfres.ψ, scfres.occupation)
        ϕ = rhs + ψ

        @testset "self-adjointness of solve_ΩplusK_split" begin
            @test isapprox(real(dot(ϕ, solve_ΩplusK_split(scfres, rhs).δψ)),
                           real(dot(solve_ΩplusK_split(scfres, ϕ).δψ, rhs)),
                           atol=1e-7)
        end

    end

end

