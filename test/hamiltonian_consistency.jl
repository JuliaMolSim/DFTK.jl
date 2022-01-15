using Test
using DFTK
import DFTK: mpi_sum

include("testcases.jl")
using Random
Random.seed!(0)

function test_consistency_term(term; rtol=1e-3, atol=1e-8, ε=1e-8, kgrid=[1, 2, 3],
                               lattice=silicon.lattice, Ecut=10, spin_polarization=:none)
    sspol = spin_polarization != :none ? " ($spin_polarization)" : ""
    @testset "Hamiltonian consistency $(typeof(term)) $sspol" begin
        n_dim = 3 - count(iszero, eachcol(lattice))
        Si = n_dim == 3 ? ElementPsp(14, psp=load_psp(silicon.psp)) : ElementCoulomb(:Si)
        atoms = [Si => silicon.positions]
        model = Model(lattice; n_electrons=silicon.n_electrons, atoms=atoms, terms=[term],
                      spin_polarization=spin_polarization)
        basis = PlaneWaveBasis(model; Ecut, kgrid, use_symmetry=false)

        n_electrons = silicon.n_electrons
        n_bands = div(n_electrons, 2, RoundUp)
        filled_occ = DFTK.filled_occupation(model)

        ψ = [Matrix(qr(randn(ComplexF64, length(G_vectors(basis, kpt)), n_bands)).Q)
             for kpt in basis.kpoints]
        occupation = [filled_occ * rand(n_bands) for _ in 1:length(basis.kpoints)]
        occ_scaling = n_electrons / sum(sum(occupation))
        occupation = [occ * occ_scaling for occ in occupation]
        ρ = compute_density(basis, ψ, occupation)

        δψ = [randn(ComplexF64, size(ψ[ik])) for ik = 1:length(basis.kpoints)]
        ψ_trial = ψ .+ ε .* δψ
        ρ_trial = compute_density(basis, ψ_trial, occupation)

        @assert length(basis.terms) == 1
        E0, ham = energy_hamiltonian(basis, ψ, occupation; ρ=ρ)
        E1, _ = energy_hamiltonian(basis, ψ_trial, occupation; ρ=ρ_trial)
        diff = (E1.total - E0.total)/ε

        diff_predicted = 0.0
        for (ik, kpt) in enumerate(basis.kpoints)
            Hψ = ham.blocks[ik]*ψ[ik]
            δψHψ = sum(occupation[ik][iband] * real(dot(δψ[ik][:, iband], Hψ[:, iband]))
                       for iband=1:n_bands)

            diff_predicted += 2 * basis.kweights[ik] * δψHψ
        end
        diff_predicted = mpi_sum(diff_predicted, basis.comm_kpts)

        err = abs(diff - diff_predicted)
        @test err < rtol * abs(E0.total) || err < atol
    end
end

test_consistency_term(Kinetic())
test_consistency_term(AtomicLocal())
test_consistency_term(AtomicNonlocal())
test_consistency_term(ExternalFromReal(X -> cos(X[1])))
test_consistency_term(ExternalFromFourier(X -> randn()))
test_consistency_term(LocalNonlinearity(ρ -> ρ^2))
test_consistency_term(Hartree())
test_consistency_term(Ewald())
test_consistency_term(PspCorrection())
test_consistency_term(Xc(:lda_xc_teter93))
test_consistency_term(Xc(:lda_xc_teter93), spin_polarization=:collinear)
test_consistency_term(Xc(:gga_x_pbe), spin_polarization=:collinear)

let
    a = 6
    pot(x, y, z) = (x - a/2)^2 + (y - a/2)^2
    Apot(x, y, z) = .2 * [y - a/2, -(x - a/2), 0]
    Apot(X) = Apot(X...)
    test_consistency_term(Magnetic(Apot);
                          kgrid=[1, 1, 1], lattice=[a 0 0; 0 a 0; 0 0 0], Ecut=20)
    test_consistency_term(DFTK.Anyonic(2, 3.2);
                          kgrid=[1, 1, 1], lattice=[a 0 0; 0 a 0; 0 0 0], Ecut=20)
end
