using Test
using DFTK
using LinearMaps
import DFTK: ortho_qr, pack_ψ, unpack_ψ, proj_tangent, proj_tangent!
import DFTK: apply_K, apply_Ω, filled_occupation
import DFTK: precondprep!, FunctionPreconditioner

include("testcases.jl")

if mpi_nprocs() == 1  # Distributed implementation not yet available
    @testset "Compute SVD of Ω+K" begin
        Ecut = 5
        tol = 1e-12
        numval = 1 # number of eigenvalues we want to compute

        Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
        model = model_LDA(silicon.lattice, [Si => silicon.positions])
        basis = PlaneWaveBasis(model, Ecut, kgrid=[1,1,1])
        N = div(model.n_electrons, filled_occupation(model))

        ρ0 = zeros(basis.fft_size..., 1)
        scfres = self_consistent_field(basis; ρ=ρ0, tol=tol)
        ψ = [scfres.ψ[1][:,1:N]]

        T = eltype(basis)
        pack(ψ) = Array(reinterpret(T, pack_ψ(basis, ψ)))
        unpack(x) = unpack_ψ(basis, reinterpret(Complex{T}, x))
        packed_proj(δx, x) = pack(proj_tangent(unpack(δx), unpack(x)))

        # preconditioner
        Pks = [PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
        for (ik, ψk) in enumerate(ψ)
            precondprep!(Pks[ik], ψk)
        end
        function f_ldiv!(x, y)
            for n in 1:numval
                δψ = unpack(y[:,n])
                proj_tangent!(δψ, ψ)
                Pδψ = [ Pks[ik] \ δψk for (ik, δψk) in enumerate(δψ)]
                proj_tangent!(Pδψ, ψ)
                x[:,n] .= pack(Pδψ)
            end
            x
        end

        # random starting point on the tangent space to avoid eigenvalue 0
        # (we get them from outisde of the tangent space)
        x0 = hcat([pack(proj_tangent([randn(Complex{eltype(basis)}, length(G_vectors(kpt)), N)
                                      for kpt in basis.kpoints], ψ)) for n in 1:numval]...)

        # mapping of the linear system on the tangent space
        function apply_jacobian(x)
            δψ = unpack(x)
            Kδψ = apply_K(basis, δψ, ψ, scfres.ρ, scfres.occupation)
            Ωδψ = apply_Ω(basis, δψ, ψ, scfres.ham)
            pack(Ωδψ + Kδψ)
        end
        J = LinearMap{T}(apply_jacobian, size(x0, 1))

        # compute smallest eigenvalue of (Ω+K) with internal routine LOBPCG of
        # DFTK
        res = lobpcg_hyper(J, x0, prec=FunctionPreconditioner(f_ldiv!),
                           tol=1e-7)

        @test res.λ[1] > 1e-3
    end
end
