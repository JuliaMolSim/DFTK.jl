using Test
using DFTK

include("testcases.jl")
@testset "Hamiltonian consistency" begin
    using Random
    Random.seed!(0)

    function test_consistency_term(term; rtol=1e-3, atol=1e-8, ε=1e-8)
        testcase = silicon
        Ecut = 10
        lattice = testcase.lattice

        Si = ElementPsp(14, psp=load_psp(testcase.psp))
        atoms = [Si => testcase.positions]
        model = Model(lattice; n_electrons=testcase.n_electrons, atoms=atoms, terms=[term])
        basis = PlaneWaveBasis(model, Ecut; kgrid=[1, 2, 3], enable_bzmesh_symmetry=false)

        n_electrons = testcase.n_electrons
        n_bands = div(n_electrons, 2)

        ψ = [Matrix(qr(randn(ComplexF64, length(G_vectors(basis.kpoints[ik])), n_bands)).Q)
             for ik in 1:length(basis.kpoints)]
        occupation = [2*rand(n_bands) for ik in 1:length(basis.kpoints)]
        occ_scaling = n_electrons / sum(sum(occupation))
        occupation = [occ * occ_scaling for occ in occupation]

        ρ = compute_density(basis, ψ, occupation)

        dψ = [randn(ComplexF64, size(ψ[ik])) for ik = 1:length(basis.kpoints)]
        ψ_trial = ψ .+ ε .* dψ
        ρ_trial = compute_density(basis, ψ_trial, occupation)

        @assert length(basis.terms) == 1
        E0, ops = ene_ops(basis.terms[1], ψ, occupation; ρ=ρ)
        E1, _ = ene_ops(basis.terms[1], ψ_trial, occupation; ρ=ρ_trial)
        diff = (E1 - E0)/ε

        diff_predicted = 0.0
        for (ik, kpt) in enumerate(basis.kpoints)
            Hψ = ops[ik]*ψ[ik]
            dψHψ = sum(occupation[ik][iband] * real(dot(dψ[ik][:, iband], Hψ[:, iband]))
                       for iband=1:n_bands)

            diff_predicted += 2 * basis.kweights[ik] * dψHψ
        end

        err = abs(diff - diff_predicted)
        @test err < rtol * abs(E0) || err < atol
    end

    test_consistency_term(Kinetic())
    test_consistency_term(AtomicLocal())
    test_consistency_term(AtomicNonlocal())
    test_consistency_term(ExternalFromReal(X -> cos(X[1])))
    test_consistency_term(ExternalFromFourier(X -> randn()))
    test_consistency_term(PowerNonlinearity(1.0, 2.0))
    test_consistency_term(Hartree())
    test_consistency_term(Ewald())
    test_consistency_term(PspCorrection())
    test_consistency_term(Xc(:lda_xc_teter93))
end
