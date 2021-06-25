using Test
using DFTK
using KrylovKit
import DFTK: ortho_qr, pack_ψ, unpack_ψ, OrthogonalizeAndProject, proj_tangent
import DFTK: apply_K, apply_Ω, filled_occupation

include("testcases.jl")

if mpi_nprocs() == 1  # Distributed implementation not yet available
    @testset "Compute SVD of Ω+K" begin
        Ecut = 5
        tol = 1e-7

        Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
        model = model_DFT(silicon.lattice, [Si => silicon.positions], [:lda_xc_teter93])
        basis = PlaneWaveBasis(model, Ecut, kgrid=[1,1,1])
        N = div(model.n_electrons, filled_occupation(model))

        ρ0 = zeros(basis.fft_size..., 1)
        scfres = self_consistent_field(basis; ρ=ρ0, tol=tol)
        ψ = [scfres.ψ[1][:,1:N]]

        # packing routines
        # some care is needed here : K is real-linear but not complex-linear because
        # of the computation of δρ from δψ. To overcome this difficulty, instead of
        # seeing the jacobian as an operator from C^Nb to C^Nb, we see it as an
        # operator from R^2Nb to R^2Nb. In practice, this is done with the
        # reinterpret function from julia. Thus, we are sure that the map we define
        # below apply_jacobian is self-adjoint.
        T = eltype(basis)
        pack(ψ) = Array(reinterpret(T, pack_ψ(basis, ψ)))
        unpack(x) = unpack_ψ(basis, reinterpret(Complex{T}, x))
        packed_proj(δx, x) = pack(proj_tangent(unpack(δx), unpack(x)))

        function jac(x)
            δψ = unpack(x)
            Kδψ = apply_K(basis, δψ, ψ, scfres.ρ, scfres.occupation)
            Ωδψ = apply_Ω(basis, δψ, ψ, scfres.ham)
            pack(Ωδψ + Kδψ)
        end

        ψ0 = proj_tangent([randn(Complex{eltype(basis)}, length(G_vectors(kpt)), N)
                           for kpt in basis.kpoints], ψ)
        egval, _ = eigsolve(jac, pack(ψ0), 1, :SR;
                            tol=1e-7, verbosity=0, eager=true,
                            orth=OrthogonalizeAndProject(packed_proj, pack(ψ)))
        @test real(egval[1]) > 0
    end
end
