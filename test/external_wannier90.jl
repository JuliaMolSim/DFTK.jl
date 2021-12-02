using Test
using DFTK
include("testcases.jl")

if !Sys.iswindows() && mpi_nprocs() == 1
@testset "Test run_wannier90" begin
    using wannier90_jll

    Si = ElementPsp(silicon.atnum, psp=load_psp("hgh/lda/Si-q4"))
    model  = model_LDA(silicon.lattice, [Si => silicon.positions])
    basis  = PlaneWaveBasis(model; Ecut=5, kgrid=[4, 4, 4])
    scfres = self_consistent_field(basis, tol=1e-12, n_bands=12)

    fileprefix = "wannier90_outputs/Si"
    run_wannier90(scfres; fileprefix,
                  n_wannier=8, bands_plot=true,
                  num_print_cycles=50, num_iter=500,
                  dis_win_max=17.185257,
                  dis_froz_max=6.8318033,
                  dis_num_iter=120,
                  dis_mix_ratio=1.0,
                  wannier_plot=true)

    @test  isfile("wannier90_outputs/Si.amn")
    @test  isfile("wannier90_outputs/Si.chk")
    @test  isfile("wannier90_outputs/Si.eig")
    @test  isfile("wannier90_outputs/Si.mmn")
    @test  isfile("wannier90_outputs/Si.nnkp")
    @test  isfile("wannier90_outputs/Si.win")
    @test  isfile("wannier90_outputs/Si.wout")
    @test !isfile("wannier90_outputs/Si.werr")

    # remove produced files
    rm("wannier90_outputs", recursive=true)
end
end
