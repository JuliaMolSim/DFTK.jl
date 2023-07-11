using Test
using DFTK
include("../testcases.jl")

if !Sys.iswindows() && mpi_nprocs() == 1
@testset "Test run_wannier" begin
    using Wannier
    model  = model_LDA(silicon.lattice, silicon.atoms, silicon.positions)
    basis  = PlaneWaveBasis(model; Ecut=5, kgrid=[4, 4, 4], kshift=[1, 1, 1]/2)
    nbandsalg = AdaptiveBands(model; n_bands_converge=12)
    scfres = self_consistent_field(basis; nbandsalg, tol=1e-12)

    fileprefix = "wannier_outputs/Si"
    run_wannier(scfres; fileprefix,
                n_wann=8, bands_plot=true,
                dis_win_max=17.185257,
                dis_froz_max=6.8318033,
                wannier_plot=true)

    @test  isfile("wannier_outputs/Si.amn")
    @test  isfile("wannier_outputs/Si.eig")
    @test  isfile("wannier_outputs/Si.mmn")
    @test  isfile("wannier_outputs/Si.win")

    # remove produced files
    rm("wannier_outputs", recursive=true)
end
end
