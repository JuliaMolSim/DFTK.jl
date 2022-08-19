using Test
using DFTK
include("testcases.jl")

@testset "Compare scf results in unit cell and supercell" begin
    Ecut = 4; kgrid = [3,3,3]; tol=1e-7; kshift=zeros(3);
    # Parameters
    Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
    model = model_LDA(silicon.lattice, [Si, Si], silicon.positions)
    basis = PlaneWaveBasis(model; Ecut, kgrid, kshift)
    basis_supercell = cell_to_supercell(basis)
    # SCF
    scfres = self_consistent_field(basis, tol=tol)
    scfres_supercell_manual = self_consistent_field(basis_supercell, tol=tol)
    scfres_supercell = cell_to_supercell(scfres)

    # Compare energies
    @test norm(scfres.energies.total*prod(kgrid) -
               scfres_supercell_manual.energies.total) < 1e-5
    @test scfres.energies.total*prod(kgrid) ≈ scfres_supercell.energies.total

    # Compare densities
    ρ_ref = DFTK.interpolate_density(dropdims(scfres.ρ, dims=4), basis, basis_supercell)
    @test norm(ρ_ref .- scfres_supercell.ρ) < 1e-5
    @test norm(ρ_ref .- scfres_supercell_manual.ρ) < 1e-3
end
