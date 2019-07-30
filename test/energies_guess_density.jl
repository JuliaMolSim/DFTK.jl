using Test
using DFTK: PlaneWaveBasis, Species, build_local_potential, build_nonlocal_projectors
using DFTK: PotHartree, PotXc, guess_gaussian_sad, empty_potential
using DFTK: update_energies_potential!, PreconditionerKinetic, lobpcg, compute_density
using DFTK: load_psp, update_energies_1e!, Functional, Hamiltonian

include("silicon_testcases.jl")

@testset "Evaluate energies of guess density" begin
    Ecut = 15
    n_bands = 8
    grid_size = [27, 27, 27]

    basis = PlaneWaveBasis(lattice, grid_size, Ecut, kpoints, kweights, ksymops)
    Si = Species(atnum, psp=load_psp("si-pade-q4.hgh"))
    occupation = [[2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0] for i in 1:length(kpoints)]

    # Build Hamiltonian and guess density
    ham = Hamiltonian(basis, pot_local=build_local_potential(basis, Si => positions),
                      pot_nonlocal=build_nonlocal_projectors(basis, Si => positions),
                      pot_hartree=PotHartree(basis),
                      pot_xc=PotXc(basis, :lda_x, :lda_c_vwn))
    ρ = guess_gaussian_sad(basis, Si => positions)

    # Run one diagonalisation
    values_hartree = empty_potential(ham.pot_hartree)
    values_xc = empty_potential(ham.pot_xc)
    energies = Dict{Symbol, real(eltype(ρ))}()
    update_energies_potential!(energies, values_hartree, ham.pot_hartree, ρ)
    update_energies_potential!(energies, values_xc, ham.pot_xc, ρ)
    @test energies[:PotHartree] ≈  0.3527293727197568  atol=5e-8
    @test energies[:lda_x]      ≈ -1.9374982215670051  atol=5e-8
    @test energies[:lda_c_vwn]  ≈ -0.3658183654888113  atol=5e-8

    prec = PreconditionerKinetic(ham, α=0.1)
    res = lobpcg(ham, n_bands, pot_hartree_values=values_hartree,
                  pot_xc_values=values_xc, prec=prec, tol=1e-9)

    # Compute energies
    ρ = compute_density(basis, res.X, occupation)
    update_energies_1e!(energies, ham, ρ, res.X, occupation)
    update_energies_potential!(energies, values_hartree, ham.pot_hartree, ρ)
    update_energies_potential!(energies, values_xc, ham.pot_xc, ρ)

    @test energies[:PotHartree]  ≈  0.6477025793366571  atol=5e-8
    @test energies[:lda_x]       ≈ -2.0625264692588163  atol=5e-8
    @test energies[:lda_c_vwn]   ≈ -0.3750064723572997  atol=5e-8
    @test energies[:PotLocal]    ≈ -2.367978663117999   atol=5e-8
    @test energies[:PotNonLocal] ≈  1.6527493682542034  atol=5e-8
    @test energies[:Kinetic]     ≈  3.291847293270256   atol=5e-8
end
