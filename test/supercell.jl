using Test
using DFTK
include("testcases.jl")

@testset "Compare scf results in unit cell and supercell" begin
    Ecut = 5; kgrid = [2,2,2]; tol=1e-7; kshift=zeros(3);
    # Parameters
    Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
    model = model_DFT(silicon.lattice, [Si => silicon.positions], [:lda_xc_teter93])
    basis = PlaneWaveBasis(model; Ecut=Ecut, kgrid=kgrid, kshift=kshift)
    basis_supercell = DFTK.cell_to_supercell(basis)
    # SCF
    scfres = self_consistent_field(basis, tol=tol)
    scfres_supercell_manual = self_consistent_field(basis_supercell, tol=tol)
    scfres_supercell = DFTK.cell_to_supercell(scfres)

    @test norm(scfres.energies.total*8 - scfres_supercell.energies.total) < 1e-5
    @test scfres.energies.total*8 == scfres_supercell.energies.total
end
