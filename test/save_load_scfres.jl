# Import statements are not all included in front to test for 
# error statements in dispatch

using DFTK

include("testcases.jl")

Si = ElementPsp(14, psp=load_psp(silicon.psp))
atoms = [Si => silicon.positions]
model = model_LDA(silicon.lattice, atoms)
kgrid = [1, 1, 1]
Ecut = 7
basis = PlaneWaveBasis(model, Ecut; kgrid = kgrid)
scfres = self_consistent_field(basis)

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


