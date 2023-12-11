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

@testitem "Test calling Wannier.Model with scfres" tags=[:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using Wannier
    silicon = TestCases.silicon

    model  = model_LDA(silicon.lattice, silicon.atoms, silicon.positions)
    basis  = PlaneWaveBasis(model; Ecut=5, kgrid=[4, 4, 4], kshift=[1, 1, 1]/2)
    nbandsalg = AdaptiveBands(model; n_bands_converge=12)
    scfres = self_consistent_field(basis; nbandsalg, tol=1e-12)

    fileprefix = "wannierjl_outputs/Si"
    wannier_model = Wannier.Model(scfres; fileprefix,
                  n_wannier=8, bands_plot=true,
                  num_print_cycles=50, num_iter=500,
                  dis_win_max=17.185257,
                  dis_froz_max=6.8318033,
                  dis_mix_ratio=1.0,
                  wannier_plot=true)

    wannier_model.U .= disentangle(wannier_model; max_iter=500)

    # for now the Wannier.jl compat writes the amn, eig, mmn and win files
    @test  isfile("wannierjl_outputs/Si.amn")
    @test !isfile("wannierjl_outputs/Si.chk")
    @test  isfile("wannierjl_outputs/Si.eig")
    @test  isfile("wannierjl_outputs/Si.mmn")
    @test !isfile("wannierjl_outputs/Si.nnkp")
    @test  isfile("wannierjl_outputs/Si.win")
    @test !isfile("wannierjl_outputs/Si.wout") # NO .wout file, the output is in the `wannier_model`
    @test !isfile("wannierjl_outputs/Si.werr")

    # remove produced files
    rm("wannierjl_outputs", recursive=true)
end

