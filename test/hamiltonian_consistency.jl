using Test
using DFTK

include("testcases.jl")
@testset "Hamiltonian consistency" begin

    testcase = silicon
    Ecut = 10
    lattice = testcase.lattice

    Si = ElementPsp(14, psp=load_psp(testcase.psp))
    atoms = [Si => testcase.positions]

    model = Model(lattice; n_electrons=testcase.n_electrons, atoms=atoms,
                  terms=[Kinetic(),
                         AtomicLocal(),
                         AtomicNonlocal(),
                         ExternalFromReal(X -> cos(X[1])),
                         ExternalFromFourier(X -> randn()),
                         PowerNonlinearity(1.0, 2.0),
                         Hartree(),
                         Ewald(),
                         PspCorrection(),
                         Xc(:lda_xc_teter93),
                         ]
                  )
    basis = PlaneWaveBasis(model, Ecut; kgrid=[1, 2, 3], enable_bzmesh_symmetry=false)

    n_electrons = testcase.n_electrons
    nbands = div(n_electrons, 2)

    using Random
    Random.seed!()

    ψ = [Matrix(qr(randn(ComplexF64, length(G_vectors(basis.kpoints[ik])), nbands)).Q) for ik in 1:length(basis.kpoints)]
    # occupation = [fill(2, nbands) for ik in 1:length(basis.kpoints)]
    occupation = [2*rand(nbands) for ik in 1:length(basis.kpoints)]

    ρ = compute_density(basis, ψ, occupation)

    ε = 1e-8
    dψ = [randn(ComplexF64, size(ψ[ik])) for ik = 1:length(basis.kpoints)]
    ψ_trial = ψ .+ ε .* dψ
    ρ_trial = compute_density(basis, ψ_trial, occupation)

    for it = 1:length(basis.terms)
        E0, ops = ene_ops(basis.terms[it], ψ, occupation; ρ=ρ)
        E1, _ = ene_ops(basis.terms[it], ψ_trial, occupation; ρ=ρ_trial)
        diff = (E1 - E0)/ε

        diff_predicted = 0.0
        for (ik, kpt) in enumerate(basis.kpoints)
            Hψ = ops[ik]*ψ[ik]
            diff_predicted += 2*basis.kweights[ik]*sum(occupation[ik][iband] * real(dot(dψ[ik][:, iband], Hψ[:, iband])) for iband=1:nbands)
        end

        err = abs(diff - diff_predicted)
        @test err < 1e-3 * abs(E0) || err < 1e-8
    end
end
