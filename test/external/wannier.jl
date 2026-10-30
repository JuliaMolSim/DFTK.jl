@testitem "Test Wannierization" tags=[:dont_test_mpi, :dont_test_windows] setup=[TestCases] begin
    using DFTK
    using Wannier
    using wannier90_jll
    using Unitful
    silicon = TestCases.silicon

    model = model_DFT(silicon.lattice, silicon.atoms, silicon.positions; functionals=LDA())
    kgrid = MonkhorstPack([4, 4, 4]; kshift=[1, 1, 1]/2)
    basis = PlaneWaveBasis(model; Ecut=5, kgrid)
    nbandsalg = AdaptiveBands(model; n_bands_converge=12)
    scfres = self_consistent_field(basis; nbandsalg, tol=1e-12)

    n_wannier = 8
    projections = [DFTK.GaussianWannierProjection(rand(3)) for _ = 1:n_wannier]

    mktempdir() do tmpdir
        @testset "Using wannier90" begin
                run_wannier90(scfres;
                    projections,
                    fileprefix="$tmpdir/Si",
                    n_bands=10, n_wannier, bands_plot=true,
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

        @testset "Using Wannier.jl agrees with wannier90" begin
            model_files  = Wannier.read_w90("$tmpdir/Si")
            model_memory = Wannier.Model(scfres;
                                          n_bands=10, n_wannier, projections,
                                          dis_froz_max=6.8318033u"eV")

            # Files store numbers to a limited number of digits
            for field in (:lattice, :atom_positions, :kpoints, :M, :U, :E, :recip_lattice)
                @test getfield(model_memory, field) ≈ getfield(model_files, field) atol=1e-5
            end
            @test model_memory.atom_labels == model_files.atom_labels
            @test model_memory.kgrid == model_files.kgrid
            @test model_memory.frozen_bands == model_files.frozen_bands
            @test any(model_memory.frozen_bands)
            @test model_memory.bvectors.kpb_k == model_files.bvectors.kpb_k
            @test model_memory.bvectors.kpb_b == model_files.bvectors.kpb_b
            @test model_memory.bvectors.weights ≈ model_files.bvectors.weights rtol=1e-5

            Ω_initial = omega(model_memory).Ω
            model_memory.U .= disentangle(model_memory; max_iter=500)
            @test omega(model_memory).Ω < Ω_initial
        end
    end
end
