using Test
using DFTK: model_dft, PlaneWaveBasis, guess_density, Element, load_psp
using DFTK: update_energies, Hamiltonian, lobpcg_hyper

include("testcases.jl")

# TODO Once we have converged SCF densities in a file it would be better to instead / also
#      test the energies of these densities and compare them directly to the reference
#      energies obtained in the data files

@testset "Evaluate energies of guess density" begin
    Ecut = 15
    n_bands = 8
    fft_size = [27, 27, 27]

    Si = Element(silicon.atnum, psp=load_psp(silicon.psp))
    model = model_dft(silicon.lattice, [:lda_x, :lda_c_vwn], [Si => silicon.positions])
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)

    ρ0 = guess_density(basis, [Si => silicon.positions])
    ham = Hamiltonian(basis, ρ0)

    e0_Hartree, _ = model.build_hartree(basis, Ref(0.0), nothing; ρ=ρ0)
    e0_XC, _ = model.build_xc(basis, Ref(0.0), nothing; ρ=ρ0)
    @test e0_Hartree[] ≈  0.3527293727197568  atol=5e-8
    @test e0_XC[]      ≈ -2.3033165870558165  atol=5e-8

    # Run one diagonalisation and compute energies
    res = diagonalise_all_kblocks(lobpcg_hyper, ham, n_bands, tol=1e-9)
    occupation = [[2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0]
                  for i in 1:length(silicon.kcoords)]
    energies = update_energies(ham, res.X, occupation)

    @test energies[:Kinetic]     ≈  3.291847293270256   atol=5e-8
    @test energies[:PotExternal] ≈ -2.367978663117999   atol=5e-8
    @test energies[:PotNonLocal] ≈  1.6527493682542034  atol=5e-8
    @test energies[:PotHartree]  ≈  0.6477025793366571  atol=5e-8
    @test energies[:PotXC]       ≈ -2.4375329416161162  atol=5e-8
end
