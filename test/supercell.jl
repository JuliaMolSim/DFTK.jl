using Test
using DFTK
include("testcases.jl")

@testset "Compare scf results in unit cell and supercell" begin
    Ecut = 5; kgrid = [2,2,2]; n_bands = 4;
    # Parameters
    Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
    model = model_DFT(silicon.lattice, [Si => silicon.positions], [:lda_xc_teter93])
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; kgrid=kgrid)
    basis_supercell = cell_to_supercell(basis)
    # SCF
    scfres = self_consistent_field(basis, n_bands=n_bands, tol=tol)
    scfres_supercell = self_consistent_field(basis_supercell,
                                             n_bands=n_bands*8,
                                             tol=tol)
    # Basic test on energies. Some ideas of other tests other than plotting ?
    @test norm(scfres.energies.total - scfres_supercell.energies.total) < 1e-3
end
