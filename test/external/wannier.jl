@testitem "Test Wannierization" tags=[:dont_test_mpi, :dont_test_windows] setup=[TestCases] begin
    using DFTK
    using Wannier
    using wannier90_jll
    silicon = TestCases.silicon

    model  = model_LDA(silicon.lattice, silicon.atoms, silicon.positions)
    basis  = PlaneWaveBasis(model; Ecut=5, kgrid=[4, 4, 4], kshift=[1, 1, 1]/2)
    nbandsalg = AdaptiveBands(model; n_bands_converge=12)
    scfres = self_consistent_field(basis; nbandsalg, tol=1e-12)

    @testset begin
        @testset "Using wannier90" begin
            mktempdir() do tmpdir
                run_wannier90(scfres;
                    fileprefix="$tmpdir/Si",
                    n_wannier=8, bands_plot=true,
                    num_print_cycles=50, num_iter=500,
                    dis_win_max=17.185257,
                    dis_froz_max=6.8318033,
                    dis_num_iter=120,
                    dis_mix_ratio=1.0,
                    wannier_plot=true)

                @test  isfile("$tmpdir/Si.amn")
                @test  isfile("$tmpdir/Si.chk")
                @test  isfile("$tmpdir/Si.eig")
                @test  isfile("$tmpdir/Si.mmn")
                @test  isfile("$tmpdir/Si.nnkp")
                @test  isfile("$tmpdir/Si.win")
                @test  isfile("$tmpdir/Si.wout")
                @test !isfile("$tmpdir/Si.werr")
            end
        end

        @testset "Using Wannier.jl" begin
            mktempdir() do tmpdir
                wannier_model = Wannier.Model(scfres;
                    fileprefix="$tmpdir/Si",
                    n_wannier=8, bands_plot=true,
                    dis_win_max=17.185257,
                    dis_froz_max=6.8318033,
                    dis_mix_ratio=1.0,
                    wannier_plot=true)

                wannier_model.U .= disentangle(wannier_model; max_iter=500)

                # for now the Wannier.jl compat writes the amn, eig, mmn and win files
                @test  isfile("$tmpdir/Si.amn")
                @test !isfile("$tmpdir/Si.chk")
                @test  isfile("$tmpdir/Si.eig")
                @test  isfile("$tmpdir/Si.mmn")
                @test !isfile("$tmpdir/Si.nnkp")
                @test  isfile("$tmpdir/Si.win")
                @test !isfile("$tmpdir/Si.wout") # NO .wout file, the output is in the `wannier_model`
                @test !isfile("$tmpdir/Si.werr")
            end
        end
    end
end
