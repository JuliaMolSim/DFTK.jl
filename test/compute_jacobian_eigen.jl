using Test
using DFTK
using LinearMaps
import DFTK: ortho_qr, pack_ψ, unpack_ψ, reinterpret_real, reinterpret_complex
import DFTK: proj_tangent, proj_tangent!
import DFTK: apply_K, apply_Ω, filled_occupation, select_occupied_orbitals
import DFTK: precondprep!, FunctionPreconditioner

include("testcases.jl")

if mpi_nprocs() == 1  # Distributed implementation not yet available

    function eigen_ΩplusK(basis::PlaneWaveBasis{T}, ψ, numval) where T
        # check that there are no virtual orbitals
        model = basis.model
        filled_occ = filled_occupation(model)
        n_spin = model.n_spin_components
        n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)
        @assert n_bands == size(ψ[1], 2)

        occupation = [filled_occ * ones(T, n_bands) for kpt = basis.kpoints]

        pack(ψ) = reinterpret_real(pack_ψ(ψ))
        unpack(x) = unpack_ψ(reinterpret_complex(x), size.(ψ))

        # compute quantites at the point which define the tangent space
        ρ = compute_density(basis, ψ, occupation)
        _, H = energy_hamiltonian(basis, ψ, occupation; ρ=ρ)

        # preconditioner
        Pks = [PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
        for (ik, ψk) in enumerate(ψ)
            precondprep!(Pks[ik], ψk)
        end
        function f_ldiv!(x, y)
            for n in 1:size(y, 2)
                δψ = unpack(y[:, n])
                proj_tangent!(δψ, ψ)
                Pδψ = [ Pks[ik] \ δψk for (ik, δψk) in enumerate(δψ)]
                proj_tangent!(Pδψ, ψ)
                x[:, n] .= pack(Pδψ)
            end
            x
        end

        # random starting point on the tangent space to avoid eigenvalue 0
        x0 = map(1:numval) do n
            initial = map(basis.kpoints) do kpt
                n_Gk = length(G_vectors(basis, kpt))
                randn(Complex{eltype(basis)}, n_Gk, n_bands)
            end
            pack(proj_tangent(initial, ψ))
        end
        x0 = hcat(x0...)

        # Rayleigh-coefficients
        Λ = [ψk'Hψk for (ψk, Hψk) in zip(ψ, H * ψ)]

        # mapping of the linear system on the tangent space
        function ΩpK(x)
            δψ = unpack(x)
            Kδψ = apply_K(basis, δψ, ψ, ρ, occupation)
            Ωδψ = apply_Ω(δψ, ψ, H, Λ)
            pack(Ωδψ + Kδψ)
        end
        J = LinearMap{T}(ΩpK, size(x0, 1))

        # compute smallest eigenvalue of Ω with LOBPCG
        lobpcg_hyper(J, x0, prec=FunctionPreconditioner(f_ldiv!), tol=1e-7)
    end

    @testset "Compute eigenvalues" begin
        @testset "Compute smallest eigenvalue of Ω" begin
            Ecut = 5
            tol = 1e-12
            numval = 3  # number of eigenvalues we want to compute

            Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
            model = model_atomic(silicon.lattice, [Si => silicon.positions])
            basis = PlaneWaveBasis(model; Ecut, kgrid=[1,1,1])

            scfres = self_consistent_field(basis; tol=tol)
            ψ = select_occupied_orbitals(basis, scfres.ψ)

            res = eigen_ΩplusK(basis, ψ, numval)
            gap = scfres.eigenvalues[1][5] - scfres.eigenvalues[1][4]

            # in the linear case, the smallest eigenvalue of Ω is the gap
            @test abs(res.λ[1] - gap) < 1e-5
            @test res.λ[1] > 1e-3
        end

        @testset "Compute smallest eigenvalue of Ω+K" begin
            Ecut = 5
            tol = 1e-12
            numval = 3  # number of eigenvalues we want to compute

            Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
            model = model_LDA(silicon.lattice, [Si => silicon.positions])
            basis = PlaneWaveBasis(model; Ecut, kgrid=[1,1,1])

            scfres = self_consistent_field(basis; tol=tol)
            ψ = select_occupied_orbitals(basis, scfres.ψ)

            res = eigen_ΩplusK(basis, ψ, numval)

            @test res.λ[1] > 1e-3
        end
    end
end
