using DFTK
using WriteVTK
using JLD2
include("testcases.jl")

Si = ElementPsp(14, psp=load_psp(silicon.psp))
atoms = [Si => silicon.positions]
model = model_LDA(silicon.lattice, atoms)
kgrid = [1, 1, 1]
Ecut = 7
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
scfres = self_consistent_field(basis)

@test_throws ErrorException save_scfres("MyVTKfile.random", scfres)
@test_throws ErrorException save_scfres("MyVTKfile", scfres)

mktempdir() do tmpdir
    vtkfile = joinpath(tmpdir, "MyVTKfile.vts")
    @test save_scfres(vtkfile, scfres) == vtkfile
    @test isfile(vtkfile) == true
    jldfile = joinpath(tmpdir, "MyJLDfile.jld2")
    save_scfres(jldfile, scfres)
    @test isfile(jldfile) == true
end
