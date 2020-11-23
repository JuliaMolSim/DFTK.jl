using Test
using DFTK

include("../../src/external/dftk_to_wannier90.jl")

# SCF loop
a = 10.26 #a.u.

lattice = a / 2*[[-1.  0. -1.];   #basis.model.lattice (in a.u.)
                 [ 0   1.  1.];
                 [ 1   1.  0.]]

Si = ElementPsp(:Si, psp=load_psp("hgh/pbe/Si-q4"))
atoms = [ Si => [zeros(3), 0.25*[-1,3,-1]] ]
model = model_PBE(lattice,atoms)

kgrid = [4,4,4] # mp grid
Ecut = 20
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid, use_symmetry=false)

scfres = self_consistent_field(basis, tol=1e-12, n_bands = 12, n_ep_extra = 0 );
ψ = scfres.ψ
n_bands = size(ψ[1],2)

dftk2wan_win_file("Si",basis,scfres,kgrid,8;
                  bands_plot=true, num_print_cycles=50, num_iter=500,
                  dis_win_max       = "17.185257d0",
                  dis_froz_max      =  "6.8318033d0",
                  dis_num_iter      =  120,
                  dis_mix_ratio     = "1d0")

dftk2wan_wannierization_files("Si",basis,scfres,8)

@testset "Test production of the win file " begin
    @test isfile("Si.win")
end

@testset "Test production of the .mmn, .amn and .eig files" begin
    @test isfile("Si.mmn")
    @test isfile("Si.amn")
    @test isfile("Si.eig")
end

rm("Si.win"); rm("Si.mmn"); rm("Si.amn"); rm("Si.eig")
