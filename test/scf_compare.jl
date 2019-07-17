using Test
using DFTK

include("silicon_testcases.jl")

@testset "Compare damped and nlsolve SCF algorithm" begin
    Ecut = 5
    n_bands = 8
    n_electrons = 8
    grid_size = [15, 15, 15]
    basis = PlaneWaveBasis(lattice, grid_size, Ecut, kpoints, kweights)

    # Construct a local pseudopotential
    hgh = load_psp("si-pade-q4.hgh")
    psp_local = build_local_potential(basis, positions,
                                      G -> DFTK.eval_psp_local_fourier(hgh, basis.recip_lattice * G))
    psp_nonlocal = PotNonLocal(basis, :Si => positions, :Si => hgh)

    # Construct a Hamiltonian (Kinetic + local psp + nonlocal psp + Hartree)
    ham = Hamiltonian(basis, pot_local=psp_local,
                      pot_nonlocal=psp_nonlocal,
                      pot_hartree=PotHartree(basis))

    # Run nlsolve without guess
    prec = PreconditionerKinetic(ham, α=0.1)
    scfnl = self_consistent_field(ham, n_bands, n_electrons, lobpcg_prec=prec,
                                  algorithm=:scf_nlsolve)
    ρ_nl = scfnl[1]

    # Run damped SCF with SAD guess
    ρ = guess_gaussian_sad(basis, :Si => positions, :Si => atnum, :Si => hgh.Zion)
    scfdamp = self_consistent_field(ham, n_bands, n_electrons, lobpcg_prec=prec, ρ=ρ,
                                    algorithm=:scf_damped, damping=0.4)
    ρ_damp = scfdamp[1]

    @test maximum(abs.(ρ_nl - ρ_damp)) < 1e-6
end
