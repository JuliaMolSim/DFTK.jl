using Test

if ( !(Sys.iswindows()) && !(Sys.isapple()) )
    using DFTK
    if ( mpi_nprocs() == 1 )

        # Classic SCF
        a = 10.26
        lattice = a / 2 * [[0 1 1.];
                           [1 0 1.];
                           [1 1 0.]]
        Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
        atoms = [Si => [ones(3)/8, -ones(3)/8]]

        model = model_LDA(lattice, atoms)
        basis = PlaneWaveBasis(model; Ecut=5, kgrid=[4, 4, 4])

        scfres = self_consistent_field(basis, tol=1e-12, n_bands = 12, n_ep_extra = 0 );
        ψ = scfres.ψ
        n_bands = size(ψ[1],2)

        # Run wannierization
        run_wannier90("testcases_WANNIER90/Si", scfres, 8;
                      bands_plot=true, num_print_cycles=50, num_iter=500,
                      dis_win_max       = "17.185257d0",
                      dis_froz_max      =  "6.8318033d0",
                      dis_num_iter      =  120,
                      dis_mix_ratio     = "1d0")

        @testset "Test production of the win file " begin
            @test isfile("testcases_WANNIER90/Si.win")
        end

        @testset "Test production of the .mmn, .amn and .eig files" begin
            @test isfile("testcases_WANNIER90/Si.mmn")
            @test isfile("testcases_WANNIER90/Si.amn")
            @test isfile("testcases_WANNIER90/Si.eig")
        end

        # remove produced files
        for output_file in filter(startswith("Si"), readdir("testcases_WANNIER90"))
            rm("testcases_WANNIER90/$(output_file)")
        end
    end
end
