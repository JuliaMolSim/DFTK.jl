using Test
using DFTK
using wannier90_jll

if( (mpi_nprocs() == 1) && !(Sys.iswindows()) )

    # Classic SCF
    a = 10.26 #a.u.

    lattice = a / 2*[[-1.  0. -1.];   #basis.model.lattice (in a.u.)
                     [ 0   1.  1.];
                     [ 1   1.  0.]]

    Si = ElementPsp(:Si, psp=load_psp("hgh/pbe/Si-q4"))
    atoms = [ Si => [zeros(3), 0.25*[-1,3,-1]] ]

    model = model_PBE(lattice,atoms)
    basis = PlaneWaveBasis(model, Ecut=5, kgrid=[4, 4, 4])

    scfres = self_consistent_field(basis, tol=1e-12, n_bands = 12, n_ep_extra = 0 );
    ψ = scfres.ψ
    n_bands = size(ψ[1],2)

    # Run wannierization with gaussian guess
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

    # run wannier90 with manual guess
    prefix = "testcases_WANNIER90/W90_guess_Si"
    wannier90() do exe
        run(`$exe -pp $prefix`)
    end
    nn_kpts, nn_num, projs = DFTK.read_nnkp(prefix)

    scfres_unfold = DFTK.unfold_bz(scfres);
    DFTK.write_eig(prefix, scfres_unfold)
    DFTK.write_amn(prefix, scfres_unfold, 8; guess="win", projs=projs)
    DFTK.write_mmn(prefix, scfres_unfold, nn_kpts, nn_num)

    @testset "Test production of the .mmn, .amn and .eig files" begin
        @test isfile("testcases_WANNIER90/W90_guess_Si.mmn")
        @test isfile("testcases_WANNIER90/W90_guess_Si.amn")
        @test isfile("testcases_WANNIER90/W90_guess_Si.eig")
    end

    # remove produced files
    for postfix in ("eig","mmn","amn","wout","nnkp")
        rm("$(prefix).$(postfix)")
    end
end
