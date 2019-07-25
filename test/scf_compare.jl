using Test
using DFTK

include("silicon_testcases.jl")

@testset "Compare damped and nlsolve SCF algorithm" begin
    Ecut = 5
    n_bands = 8
    grid_size = [15, 15, 15]

    basis = PlaneWaveBasis(lattice, grid_size, Ecut, kpoints, kweights)
    Si = Species(atnum, psp=load_psp("si-pade-q4.hgh"))
    n_electrons = length(positions) * n_elec_valence(Si)

    ham = Hamiltonian(basis, pot_local=build_local_potential(basis, Si => positions),
                      pot_nonlocal=build_nonlocal_projectors(basis, Si => positions),
                      pot_hartree=PotHartree(basis))

    # Run nlsolve without guess
    prec = PreconditionerKinetic(ham, α=0.1)
    scfnl = self_consistent_field(ham, n_bands, n_electrons, lobpcg_prec=prec,
                                  algorithm=:scf_nlsolve)
    ρ_nl = scfnl[1]

    # Run damped SCF with SAD guess
    ρ = guess_gaussian_sad(basis, Si => positions)
    scfdamp = self_consistent_field(ham, n_bands, n_electrons, lobpcg_prec=prec, ρ=ρ,
                                    algorithm=:scf_damped, damping=0.4)
    ρ_damp = scfdamp[1]

    @test maximum(abs.(ρ_nl - ρ_damp)) < 1e-6
end
