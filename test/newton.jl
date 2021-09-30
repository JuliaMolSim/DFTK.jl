using Test
using DFTK
import DFTK: compute_projected_gradient, solve_ΩplusK, filled_occupation

include("testcases.jl")

@testset "Newton" begin
    Ecut = 3
    fft_size = [9, 9, 9]
    Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
    model = model_DFT(silicon.lattice, [Si => silicon.positions], [:lda_xc_teter93])
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)
    scfres_start = self_consistent_field(basis, maxiter=1)

    @testset "self-adjointness of solve_ΩplusK" begin
        ψ = DFTK.select_occupied_orbitals(basis, scfres_start.ψ)

        occupation = scfres_start.occupation
        filled_occ = filled_occupation(model)
        n_spin = model.n_spin_components
        n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)
        @assert n_bands == size(ψ[1], 2)
        # number of kpoints and occupation
        Nk = length(basis.kpoints)
        occupation = [filled_occ * ones(n_bands) for ik = 1:Nk]

        rhs = compute_projected_gradient(basis, ψ, occupation)
        ϕ = rhs + ψ
        @test isapprox(
            real(dot(ϕ, solve_ΩplusK(basis, ψ, rhs, occupation))),
            real(dot(solve_ΩplusK(basis, ψ, ϕ, occupation), rhs)),
            atol=1e-7
        )
    end
end
