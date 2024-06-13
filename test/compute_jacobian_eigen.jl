# Distributed implementation not yet available
@testsetup module CompJacEig
using DFTK
using DFTK: ortho_qr, pack_ψ, unpack_ψ, reinterpret_real, reinterpret_complex
using DFTK: proj_tangent, proj_tangent!
using DFTK: apply_K, apply_Ω
using DFTK: precondprep!, FunctionPreconditioner
using LinearMaps

function eigen_ΩplusK(basis::PlaneWaveBasis{T}, ψ, occupation, numval) where {T}
    n_components = basis.model.n_components
    @assert n_components == 1

    pack(ψ) = reinterpret_real(pack_ψ(ψ))
    unpack(x) = unpack_ψ(reinterpret_complex(x), size.(ψ))

    # compute quantites at the point which define the tangent space
    ρ = compute_density(basis, ψ, occupation)
    H = energy_hamiltonian(basis, ψ, occupation; ρ).ham

    # preconditioner
    Pks = [PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
    for ik = 1:length(Pks)
        precondprep!(Pks[ik], to_composite_σG(basis, basis.kpoints[ik], ψ[ik]))
    end
    function f_ldiv!(x, y)
        for n = 1:size(y, 2)
            δψ = unpack(y[:, n])
            proj_tangent!(δψ, basis, ψ)
            Pδψ = [Pks[ik] \ δψk for (ik, δψk) in enumerate(δψ)]
            proj_tangent!(Pδψ, basis, ψ)
            x[:, n] .= pack(Pδψ)
        end
        x
    end

    # random starting point on the tangent space to avoid eigenvalue 0
    x0 = map(1:numval) do _
        initial = map(enumerate(basis.kpoints)) do (ik, kpt)
            n_Gk = length(G_vectors(basis, kpt))
            randn(Complex{T}, n_components, n_Gk, size(ψ[ik], 3))
        end
        pack(proj_tangent(basis, initial, ψ))
    end
    x0 = stack(x0)

    # Rayleigh-coefficients
    Λ = map(enumerate(zip(ψ, H * ψ))) do (ik, (ψk, Hψk))
        ψk = to_composite_σG(basis, basis.kpoints[ik], ψk)
        Hψk = to_composite_σG(basis, basis.kpoints[ik], Hψk)
        from_composite_σG(basis, basis.kpoints[ik], ψk'Hψk)
    end

    # mapping of the linear system on the tangent space
    function ΩpK(x)
        δψ = unpack(x)
        Kδψ = apply_K(basis, δψ, ψ, ρ, occupation)
        Ωδψ = apply_Ω(δψ, ψ, H, Λ)
        pack(Ωδψ + Kδψ)
    end
    J = LinearMap{T}(ΩpK, size(x0, 1))

    # compute smallest eigenvalue of Ω with LOBPCG
    lobpcg_hyper(J, x0; prec=FunctionPreconditioner(f_ldiv!), tol=1e-7)
end
end


@testitem "Compute eigenvalues" tags=[:dont_test_mpi] setup=[CompJacEig, TestCases] begin
    using DFTK
    using DFTK: select_occupied_orbitals
    using .CompJacEig: eigen_ΩplusK
    testcase = TestCases.silicon

    @testset "Compute smallest eigenvalue of Ω" begin
        numval = 3  # number of eigenvalues we want to compute

        model  = model_atomic(testcase.lattice, testcase.atoms, testcase.positions)
        basis  = PlaneWaveBasis(model; Ecut=5, kgrid=[1, 1, 1])
        scfres = self_consistent_field(basis; tol=1e-8)
        ψ, occupation = select_occupied_orbitals(basis, scfres.ψ, scfres.occupation)

        res = eigen_ΩplusK(basis, ψ, occupation, numval)
        gap = scfres.eigenvalues[1][5] - scfres.eigenvalues[1][4]

        # in the linear case, the smallest eigenvalue of Ω is the gap
        @test abs(res.λ[1] - gap) < 1e-5
        @test res.λ[1] > 1e-3
    end

    @testset "Compute smallest eigenvalue of Ω+K" begin
        numval = 3  # number of eigenvalues we want to compute

        model  = model_LDA(testcase.lattice, testcase.atoms, testcase.positions)
        basis  = PlaneWaveBasis(model; Ecut=5, kgrid=[1, 1, 1])
        scfres = self_consistent_field(basis; tol=1e-8)
        ψ, occupation = select_occupied_orbitals(basis, scfres.ψ, scfres.occupation)

        res = eigen_ΩplusK(basis, ψ, occupation, numval)
        @test res.λ[1] > 1e-3
    end
end
