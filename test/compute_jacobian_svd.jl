using Test
using DFTK
import DFTK: ortho_qr, pack_ψ, unpack_ψ, OrthogonalizeAndProject, proj_tangent
import DFTK: apply_K, apply_Ω, filled_occupation

include("testcases.jl")

@testset "Compute SVD of Ω+K" begin
    Ecut = 3
    tol = 1e-7

    Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
    model = model_DFT(silicon.lattice, [Si => silicon.positions], [:lda_xc_teter93])
    basis = PlaneWaveBasis(model, Ecut, kgrid=[1,1,1])
    N = div(model.n_electrons, filled_occupation(model))

    ρ0 = zeros(basis.fft_size..., 1)
    scfres = self_consistent_field(basis; ρ=ρ0, tol=tol)
    ψ = [scfres.ψ[1][:,1:N]]

    # packing routines
    pack(ψ) = pack_ψ(basis, ψ)
    unpack(x) = unpack_ψ(basis, x)
    packed_proj(δx, x) = pack(proj_tangent(unpack(δx), unpack(x)))

    function jac(x, flag)
        δψ = unpack(x)
        δψ = proj_tangent(δψ, ψ)
        Kδψ = apply_K(basis, δψ, ψ, scfres.ρ, scfres.occupation)
        Ωδψ = apply_Ω(basis, δψ, ψ, scfres.ham)
        pack(Ωδψ + Kδψ)
    end

    ψ0 = [ortho_qr(randn(Complex{eltype(basis)}, length(G_vectors(kpt)), N))
          for kpt in basis.kpoints]
    svd, _ = svdsolve(jac, pack(ψ0), 1, :SR;
                      tol=1e-5, verbosity=0, eager=true,
                      orth=OrthogonalizeAndProject(packed_proj, pack(ψ)))
    @test svd[1] > 0
end
