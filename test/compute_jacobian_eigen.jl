using Test
using DFTK
using LinearMaps
import DFTK: ortho_qr, pack_ψ, unpack_ψ, proj_tangent, proj_tangent!
import DFTK: apply_K, apply_Ω, filled_occupation
import DFTK: precondprep!, FunctionPreconditioner

include("testcases.jl")

@testset "Compute eigenvalues" begin
    if mpi_nprocs() == 1  # Distributed implementation not yet available
        @testset "Compute smallest eigenvalue of Ω" begin
            Ecut = 5
            tol = 1e-12
            numval = 1 # number of eigenvalues we want to compute

            Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
            model = model_atomic(silicon.lattice, [Si => silicon.positions])
            basis = PlaneWaveBasis(model, Ecut, kgrid=[1,1,1])
            n_bands = div(model.n_electrons, filled_occupation(model))

            ρ0 = zeros(basis.fft_size..., 1)
            scfres = self_consistent_field(basis; ρ=ρ0, tol=tol)
            ψ = [scfres.ψ[1][:,1:n_bands]]

            T = eltype(basis)
            pack(ψ) = Array(reinterpret(T, pack_ψ(basis, ψ)))
            unpack(x) = unpack_ψ(basis, reinterpret(Complex{T}, x))

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
            x0 = hcat([pack(proj_tangent([randn(Complex{eltype(basis)}, length(G_vectors(kpt)), n_bands)
                                          for kpt in basis.kpoints], ψ)) for n in 1:numval]...)

            # Rayleigh-coefficients
            Λ = map(enumerate(ψ)) do (ik, ψk)
                Hk = scfres.ham.blocks[ik]
                Hψk = Hk * ψk
                ψk'Hψk
            end

            # mapping of the linear system on the tangent space
            function Ω(x)
                δψ = unpack(x)
                Ωδψ = apply_Ω(δψ, ψ, scfres.ham, Λ)
                pack(Ωδψ)
            end
            J = LinearMap{T}(Ω, size(x0, 1))

            # compute smallest eigenvalue of (Ω+K) with LOBPCG
            res = lobpcg_hyper(J, x0, prec=FunctionPreconditioner(f_ldiv!), tol=1e-7)
            gap = scfres.eigenvalues[1][5] - scfres.eigenvalues[1][4]

            # in the linear case, the smallest eigenvalue of Ω is the gap
            @test abs(res.λ[1] - gap) < 1e-5
            @test res.λ[1] > 1e-3
        end

        @testset "Compute smallest eigenvalue of Ω+K" begin
            Ecut = 5
            tol = 1e-12
            numval = 1 # number of eigenvalues we want to compute

            Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
            model = model_LDA(silicon.lattice, [Si => silicon.positions])
            basis = PlaneWaveBasis(model, Ecut, kgrid=[1,1,1])
            n_bands = div(model.n_electrons, filled_occupation(model))

            ρ0 = zeros(basis.fft_size..., 1)
            scfres = self_consistent_field(basis; ρ=ρ0, tol=tol)
            ψ = [scfres.ψ[1][:,1:n_bands]]

            T = eltype(basis)
            pack(ψ) = Array(reinterpret(T, pack_ψ(basis, ψ)))
            unpack(x) = unpack_ψ(basis, reinterpret(Complex{T}, x))

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
            x0 = hcat([pack(proj_tangent([randn(Complex{eltype(basis)}, length(G_vectors(kpt)), n_bands)
                                          for kpt in basis.kpoints], ψ)) for n in 1:numval]...)

            # Rayleigh-coefficients
            Λ = map(enumerate(ψ)) do (ik, ψk)
                Hk = scfres.ham.blocks[ik]
                Hψk = Hk * ψk
                ψk'Hψk
            end

            # mapping of the linear system on the tangent space
            function ΩpK(x)
                δψ = unpack(x)
                Kδψ = apply_K(basis, δψ, ψ, scfres.ρ, scfres.occupation)
                Ωδψ = apply_Ω(δψ, ψ, scfres.ham, Λ)
                pack(Ωδψ + Kδψ)
            end
            J = LinearMap{T}(ΩpK, size(x0, 1))

            # compute smallest eigenvalue of (Ω+K) with LOBPCG
            res = lobpcg_hyper(J, x0, prec=FunctionPreconditioner(f_ldiv!), tol=1e-7)

            @test res.λ[1] > 1e-3
        end
    end
end
