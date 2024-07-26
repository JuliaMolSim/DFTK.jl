@testmodule ScfresAgreement begin
using Test
using DFTK

function test_scfres_agreement(tested, ref; test_ψ=true)
    @test tested.basis.model.lattice           == ref.basis.model.lattice
    @test tested.basis.model.temperature       == ref.basis.model.temperature
    @test tested.basis.model.smearing          == ref.basis.model.smearing
    @test tested.basis.model.εF                == ref.basis.model.εF
    @test tested.basis.model.symmetries        == ref.basis.model.symmetries
    @test tested.basis.model.spin_polarization == ref.basis.model.spin_polarization

    @test tested.basis.model.positions == ref.basis.model.positions
    @test atomic_symbol.(tested.basis.model.atoms) == atomic_symbol.(ref.basis.model.atoms)

    @test tested.basis.Ecut      == ref.basis.Ecut
    @test tested.basis.kweights  == ref.basis.kweights
    @test tested.basis.fft_size  == ref.basis.fft_size
    @test tested.basis.kgrid     == ref.basis.kgrid

    kcoords_test = getproperty.(tested.basis.kpoints, :coordinate)
    kcoords_ref  = getproperty.(ref.basis.kpoints, :coordinate)
    @test kcoords_test == kcoords_ref

    # Note: Apart from the energy (which is recomputed on loading) the other quantities
    #       should be exactly as stored. For MPI runs it is possible that the density differs
    #       slightly for non-master processes as the version from the master process is stored.
    @test tested.n_iter         == ref.n_iter
    @test tested.energies.total ≈  ref.energies.total atol=1e-13
    @test tested.eigenvalues    == ref.eigenvalues
    @test tested.occupation     == ref.occupation
    @test tested.ρ              ≈  ref.ρ rtol=1e-14

    if test_ψ
        @test tested.ψ == ref.ψ
    end
end
end

@testitem "SCF checkpointing" setup=[ScfresAgreement, TestCases] tags=[:serialisation] begin
    using DFTK
    using DFTK: ScfDefaultCallback, ScfSaveCheckpoints
    using JLD2  # needed for ScfSaveCheckpoints
    using MPI
    using LinearAlgebra
    o2molecule = TestCases.o2molecule

    magnetic_moments = [1., 1.]
    model = model_PBE(o2molecule.lattice, o2molecule.atoms, o2molecule.positions;
                      temperature=0.02, smearing=Smearing.Gaussian(),
                      magnetic_moments, symmetries=false)

    kgrid = [1, mpi_nprocs(), 1]   # Ensure at least 1 kpt per process
    basis  = PlaneWaveBasis(model; Ecut=4, kgrid)
    ρ = guess_density(basis, magnetic_moments)

    # Run SCF and do checkpointing along the way
    mktempdir() do tmpdir
        filename = joinpath(tmpdir, "scfres.jld2")
        filename = MPI.bcast(filename, 0, MPI.COMM_WORLD)  # master -> everyone
        kwargs = kwargs_scf_checkpoints(basis; filename, ρ)
        nbandsalg = FixedBands(; n_bands_converge=20)
        scfres = self_consistent_field(basis; tol=1e-2, nbandsalg, kwargs...)

        kwargs = kwargs_scf_checkpoints(basis; filename, ρ)
        @test norm(kwargs.ρ - scfres.ρ) < 1e-12
        scfres = self_consistent_field(basis; tol=5e-2, nbandsalg, kwargs...)
        @test scfres.n_iter ≤ 3

        ScfresAgreement.test_scfres_agreement(scfres, load_scfres(filename); test_ψ=false)
    end
end

@testitem "Serialisation" setup=[ScfresAgreement, DictAgreement, TestCases] #=
                       =# tags=[:serialisation] begin
    using DFTK
    using JLD2
    using JSON3
    using MPI
    using Test
    using WriteVTK
    testcase = TestCases.silicon

    function test_serialisation(testcase, label; modelargs=(; ),
                                basisargs=(; Ecut=5, kgrid=(2, 3, 4)))
        model = model_LDA(testcase.lattice, testcase.atoms, testcase.positions; modelargs...)

        basis = PlaneWaveBasis(model; basisargs...)
        nbandsalg = FixedBands(; n_bands_converge=20)
        scfres = self_consistent_field(basis; tol=1e-1, nbandsalg)

        @test_throws ErrorException save_scfres("MyVTKfile.random", scfres)
        @test_throws ErrorException save_scfres("MyVTKfile", scfres)

        @testset "JLD2 ($label)" begin
            mktempdir() do tmpdir
                dumpfile = MPI.bcast(joinpath(tmpdir, "scfres.jld2"), 0, MPI.COMM_WORLD)
                save_scfres(dumpfile, scfres)
                @test isfile(dumpfile)
                ScfresAgreement.test_scfres_agreement(scfres, load_scfres(dumpfile))
                ScfresAgreement.test_scfres_agreement(scfres, load_scfres(dumpfile, basis))
            end
        end

        @testset "VTK ($label)" begin
            mktempdir() do tmpdir
                dumpfile = MPI.bcast(joinpath(tmpdir, "scfres.vts"), 0, MPI.COMM_WORLD)
                save_scfres(dumpfile, scfres; save_ψ=true)
                @test isfile(dumpfile)
            end
        end

        @testset "JSON ($label)" begin
            mktempdir() do tmpdir
                dumpfile = MPI.bcast(joinpath(tmpdir, "scfres.json"), 0, MPI.COMM_WORLD)
                save_scfres(dumpfile, scfres)
                @test isfile(dumpfile)
                data = open(JSON3.read, dumpfile)  # Get data back as dict
                DictAgreement.test_agreement_scfres(scfres, data;
                                                    test_ψ=false, explicit_reshape=true)
            end
        end
    end

    test_serialisation(testcase, "nospin notemp"; modelargs=(; spin_polarization=:none))
    test_serialisation(testcase, "collinear temp";
                       modelargs=(; spin_polarization=:collinear, temperature=0.01))
    test_serialisation(testcase, "fixed Fermi";
                       modelargs=(; εF=0.5, disable_electrostatics_check=true,
                                  temperature=1e-3))
end
