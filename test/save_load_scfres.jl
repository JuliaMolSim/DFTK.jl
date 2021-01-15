# This test is based on example docs/src/guide/tutorial.jl
# Import statements are not all included in front to test for 
# error statements in dispatch

using DFTK
using Unitful
using UnitfulAtomic

a = 5.431u"angstrom"
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]
model = model_LDA(lattice, atoms)
kgrid = [4, 4, 4]
Ecut = 7
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
scfres = self_consistent_field(basis, tol=1e-8)

@test_throws ErrorException save_scfres("MyVTKfile.vts", scfres)
@test_throws ErrorException save_scfres("MyVTKfile.jld", scfres)
@test_throws ErrorException save_scfres("MyVTKfile.random", scfres)
@test_throws ErrorException save_scfres("MyVTKfile", scfres)

using WriteVTK
using JLD2

mktempdir() do tmpdir
    cd(tmpdir)
    @test save_scfres("MyVTKfile.vts",scfres) == "MyVTKfile.vts"
    @test isfile("MyVTKfile.vts") == true
    save_scfres("MyVTKfile.jld",scfres)
    @test isfile("MyVTKfile.jld") == true
end


