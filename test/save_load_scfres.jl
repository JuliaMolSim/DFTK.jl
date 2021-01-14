# This test is based on example examples/collinear_magnetism.jl
# Import statements are not all included in front to test for 
# error statements in dispatch

using DFTK

a = 5.42352
lattice = a / 2 * [[-1  1  1];
                   [ 1 -1  1];
                   [ 1  1 -1]]
Fe = ElementPsp(:Fe, psp=load_psp("hgh/lda/Fe-q8.hgh"))
atoms = [Fe => [zeros(3)]];
kgrid = [3, 3, 3]
Ecut = 15
magnetic_moments = [Fe => [4, ]];
model = model_LDA(lattice, atoms, magnetic_moments=magnetic_moments, temperature=0.01)
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
ρspin = guess_spin_density(basis, magnetic_moments)
scfres = self_consistent_field(basis, tol=1e-6, ρspin=ρspin, mixing=KerkerMixing());

@test_throws ErrorException save_scfres("MyVTKfile.vts",scfres)
@test_throws ErrorException save_scfres("MyVTKfile.jld",scfres)
@test_throws ErrorException save_scfres("MyVTKfile.random",scfres)
@test_throws ErrorException save_scfres("MyVTKfile",scfres)

using WriteVTK
using JLD2

mktempdir() do tmpdir
    cd(tmpdir)
    @test save_scfres("MyVTKfile.vts",scfres) == "MyVTKfile.vts"
    @test isfile("MyVTKfile.vts") == true
    save_scfres("MyVTKfile.jld",scfres)
    @test isfile("MyVTKfile.jld") == true
end


