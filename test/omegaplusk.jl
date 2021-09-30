using Test
using DFTK
import DFTK: solve_ΩplusK, apply_Ω, apply_K
import DFTK: filled_occupation, compute_projected_gradient

include("testcases.jl")

@testset "Newton" begin
    Ecut = 3
    fft_size = [9, 9, 9]
    Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
    model = model_DFT(silicon.lattice, [Si => silicon.positions], [:lda_xc_teter93])
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)
    scfres_start = self_consistent_field(basis, maxiter=1)

    ψ = DFTK.select_occupied_orbitals(basis, scfres_start.ψ)

    occupation = scfres_start.occupation
    filled_occ = filled_occupation(model)
    n_spin = model.n_spin_components
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)
    @assert n_bands == size(ψ[1], 2)
    # number of kpoints and occupation
    Nk = length(basis.kpoints)
    occupation = [filled_occ * ones(n_bands) for ik = 1:Nk]

    ρ = compute_density(basis, ψ, occupation)

    rhs = compute_projected_gradient(basis, ψ, occupation)
    ϕ = rhs + ψ

    @testset "self-adjointness of solve_ΩplusK" begin 
        @test isapprox(
            real(dot(ϕ, solve_ΩplusK(basis, ψ, rhs, occupation))),
            real(dot(solve_ΩplusK(basis, ψ, ϕ, occupation), rhs)),
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
end
