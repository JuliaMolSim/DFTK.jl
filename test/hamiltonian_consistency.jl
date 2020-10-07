using Test
using DFTK

include("testcases.jl")
@testset "Hamiltonian consistency" begin
    using Random
    Random.seed!(0)

    function test_consistency_term(term; rtol=1e-3, atol=1e-8, ε=1e-8, kgrid=[1, 2, 3],
                                   lattice=silicon.lattice, Ecut=10)
        Si = ElementPsp(14, psp=load_psp(silicon.psp))
        atoms = [Si => silicon.positions]
        model = Model(lattice; n_electrons=silicon.n_electrons, atoms=atoms, terms=[term])
        basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid, use_symmetry=false)

        n_electrons = silicon.n_electrons
        n_bands = div(n_electrons, 2)

        ψ = [Matrix(qr(randn(ComplexF64, length(G_vectors(basis.kpoints[ik])), n_bands)).Q)
             for ik in 1:length(basis.kpoints)]
        occupation = [2*rand(n_bands) for ik in 1:length(basis.kpoints)]
        occ_scaling = n_electrons / sum(sum(occupation))
        occupation = [occ * occ_scaling for occ in occupation]

        ρ, ρspin = compute_density(basis, ψ, occupation)

        dψ = [randn(ComplexF64, size(ψ[ik])) for ik = 1:length(basis.kpoints)]
        ψ_trial = ψ .+ ε .* dψ
        ρ_trial, ρspin_trial = compute_density(basis, ψ_trial, occupation)

        @assert length(basis.terms) == 1
        E0, ham = energy_hamiltonian(basis, ψ, occupation; ρ=ρ, ρspin=ρspin)
        E1, _ = energy_hamiltonian(basis, ψ_trial, occupation; ρ=ρ_trial, ρspin=ρspin_trial)
        diff = (E1.total - E0.total)/ε

        diff_predicted = 0.0
        for (ik, kpt) in enumerate(basis.kpoints)
            Hψ = ham.blocks[ik]*ψ[ik]
            dψHψ = sum(occupation[ik][iband] * real(dot(dψ[ik][:, iband], Hψ[:, iband]))
                       for iband=1:n_bands)

            diff_predicted += 2 * basis.kweights[ik] * dψHψ
        end

        err = abs(diff - diff_predicted)
        @test err < rtol * abs(E0.total) || err < atol
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

    a = 6
    pot(x, y, z) = (x - a/2)^2 + (y - a/2)^2
    Apot(x, y, z) = .2 * [y - a/2, -(x - a/2), 0]
    Apot(X) = Apot(X...)
    test_consistency_term(Magnetic(Apot); kgrid=[1, 1, 1], lattice=[a 0 0; 0 a 0; 0 0 0], Ecut=20)
    test_consistency_term(DFTK.Anyonic(1); kgrid=[1, 1, 1], lattice=[a 0 0; 0 a 0; 0 0 0], Ecut=20)
end
