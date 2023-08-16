using DFTK
using JLD2
using JSON3
using WriteVTK
using MPI
import DFTK: ScfDefaultCallback, ScfSaveCheckpoints
using Test
include("testcases.jl")

function test_scfres_agreement(tested, ref)
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
    @test tested.ψ              == ref.ψ
    @test tested.ρ              ≈  ref.ρ rtol=1e-14
end


@testset "SCF checkpointing" begin
    model = model_PBE(o2molecule.lattice, o2molecule.atoms, o2molecule.positions;
                      temperature=0.02, smearing=Smearing.Gaussian(),
                      magnetic_moments=[1., 1.], symmetries=false)

    kgrid = [1, mpi_nprocs(), 1]   # Ensure at least 1 kpt per process
    basis  = PlaneWaveBasis(model; Ecut=4, kgrid)

    # Run SCF and do checkpointing along the way
    mktempdir() do tmpdir
        checkpointfile = joinpath(tmpdir, "scfres.jld2")
        checkpointfile = MPI.bcast(checkpointfile, 0, MPI.COMM_WORLD)  # master -> everyone

        callback  = ScfDefaultCallback() ∘ ScfSaveCheckpoints(checkpointfile; keep=true)
        nbandsalg = FixedBands(; n_bands_converge=20)
        scfres = self_consistent_field(basis; tol=1e-2, nbandsalg, callback)
        test_scfres_agreement(scfres, load_scfres(checkpointfile))
    end
end

function test_serialisation(label;
                            modelargs=(; spin_polarization=:collinear, temperature=0.01),
                            basisargs=(; Ecut=5, kgrid=(2, 3, 4)))
    model = model_LDA(silicon.lattice, silicon.atoms, silicon.positions; modelargs...)
    basis = PlaneWaveBasis(model; basisargs...)
    nbandsalg = FixedBands(; n_bands_converge=20)
    scfres = self_consistent_field(basis; tol=1e-1, nbandsalg)

    @test_throws ErrorException save_scfres("MyVTKfile.random", scfres)
    @test_throws ErrorException save_scfres("MyVTKfile", scfres)

    @testset "JLD2 ($label)" begin
        mktempdir() do tmpdir
            dumpfile = joinpath(tmpdir, "scfres.jld2")
            dumpfile = MPI.bcast(dumpfile, 0, MPI.COMM_WORLD)  # master -> everyone

            save_scfres(dumpfile, scfres)
            @test isfile(dumpfile)
            test_scfres_agreement(scfres, load_scfres(dumpfile))
        end
    end

    @testset "VTK ($label)" begin
        mktempdir() do tmpdir
            dumpfile = joinpath(tmpdir, "scfres.vts")
            dumpfile = MPI.bcast(dumpfile, 0, MPI.COMM_WORLD)  # master -> everyone

            save_scfres(dumpfile, scfres; save_ψ=true)
            @test isfile(dumpfile)
        end
    end

    @testset "JSON ($label)" begin
        mktempdir() do tmpdir
            dumpfile = joinpath(tmpdir, "scfres.json")
            dumpfile = MPI.bcast(dumpfile, 0, MPI.COMM_WORLD)  # master -> everyone

            save_scfres(dumpfile, scfres)
            @test isfile(dumpfile)

            data = open(JSON3.read, dumpfile)  # Get data back as dict
            # TODO test json agreement
            @test data["energies"]["total"] == scfres.energies.total
        end
    end
end

@testset "Serialisation" begin
    test_serialisation("nospin notemp")
    test_serialisation("collinear temp";
                       modelargs=(; spin_polarization=:collinear, temperature=0.01))
    test_serialisation("fixed Fermi";
                       modelargs=(; εF=0.5, disable_electrostatics_check=true,
                                  temperature=1e-3))
end
