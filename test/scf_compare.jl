using Test
using DFTK

include("silicon_testcases.jl")

@testset "Compare different SCF algorithms" begin
    Ecut = 2
    n_bands = 6
    grid_size = [9, 9, 9]

    basis = PlaneWaveBasis(lattice, grid_size, Ecut, kpoints, kweights, ksymops)
    Si = Species(atnum, psp=load_psp("si-pade-q4.hgh"))
    n_electrons = length(positions) * n_elec_valence(Si)

    ham = Hamiltonian(basis, pot_local=build_local_potential(basis, Si => positions),
                      pot_nonlocal=build_nonlocal_projectors(basis, Si => positions),
                      pot_hartree=PotHartree(basis))
    prec = PreconditionerKinetic(ham, α=0.1)

    # Run nlsolve without guess
    scfnl = self_consistent_field(ham, n_bands, n_electrons, lobpcg_prec=prec,
                                  solver=scf_nlsolve_solver())
    ρ_nl = scfnl[1]

    # Test density scaling
    scfnl = self_consistent_field(ham, n_bands, n_electrons, lobpcg_prec=prec,
                                  solver=scf_nlsolve_solver(), den_scaling=.5)
    ρ_den_scaling = scfnl[1]
    @test maximum(abs.(ρ_den_scaling - ρ_nl)) < 1e-4

    # Run damped SCF with SAD guess
    ρ = guess_gaussian_sad(basis, Si => positions)
    for solver in (scf_nlsolve_solver, scf_damping_solver, scf_anderson_solver,
                   scf_CROP_solver)
        println("Testing $solver")
        scf_res = self_consistent_field(ham, n_bands, n_electrons, lobpcg_prec=prec, ρ=ρ,
                                        solver=solver())
        ρ_alg = scf_res[1]
        @test maximum(abs.(ρ_alg - ρ_nl)) < 1e-6
    end
end
