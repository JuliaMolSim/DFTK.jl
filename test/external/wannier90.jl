@testitem "Test run_wannier90" tags=[:dont_test_mpi, :dont_test_windows] setup=[TestCases] begin
    using DFTK
    using wannier90_jll
    silicon = TestCases.silicon

    model  = model_LDA(silicon.lattice, silicon.atoms, silicon.positions)
    basis  = PlaneWaveBasis(model; Ecut=5, kgrid=[4, 4, 4], kshift=[1, 1, 1]/2)
    nbandsalg = AdaptiveBands(model; n_bands_converge=12)
    scfres = self_consistent_field(basis; nbandsalg, tol=1e-12)

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
