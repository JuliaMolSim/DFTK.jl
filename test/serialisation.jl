using DFTK
using JLD2
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
    @test tested.ρ              ≈  ref.ρ     rtol=1e-14
end


function test_checkpointing(; εF=nothing)
    label = !isnothing(εF) ? "  εF" : "none"
    @testset "$label" begin
        model = model_PBE(o2molecule.lattice, o2molecule.atoms, o2molecule.positions;
                          temperature=0.02, smearing=Smearing.Gaussian(), εF,
                          magnetic_moments=[1., 1.], symmetries=false,
                          disable_electrostatics_check=true)

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
end

function test_serialisation(; εF=nothing)
    label = !isnothing(εF) ? "  εF" : "none"
    @testset "$label" begin
        model = model_LDA(silicon.lattice, silicon.atoms, silicon.positions;
                          spin_polarization=:collinear, temperature=0.01, εF=0.5,
                          disable_electrostatics_check=true)
        kgrid = [2, 3, 4]
        basis = PlaneWaveBasis(model; Ecut=5, kgrid)
        nbandsalg = FixedBands(; n_bands_converge=20)
        scfres = self_consistent_field(basis; tol=1e-2, nbandsalg)

        @test_throws ErrorException save_scfres("MyVTKfile.random", scfres)
        @test_throws ErrorException save_scfres("MyVTKfile", scfres)

        @testset "JLD2" begin
            mktempdir() do tmpdir
                dumpfile = joinpath(tmpdir, "scfres.jld2")
                dumpfile = MPI.bcast(dumpfile, 0, MPI.COMM_WORLD)  # master -> everyone

                save_scfres(dumpfile, scfres)
                @test isfile(dumpfile)
                test_scfres_agreement(scfres, load_scfres(dumpfile))
            end
        end

        @testset "VTK" begin
            mktempdir() do tmpdir
                dumpfile = joinpath(tmpdir, "scfres.vts")
                dumpfile = MPI.bcast(dumpfile, 0, MPI.COMM_WORLD)  # master -> everyone

                save_scfres(dumpfile, scfres; save_ψ=true)
                @test isfile(dumpfile)
            end
        end
    end
end

@testset "Test checkpointing" begin
    for εF in (nothing, 0.5)
        test_checkpointing(; εF)
    end
end

@testset "Test serialisation" begin
    for εF in (nothing, 0.5)
        test_serialisation(; εF)
    end
end
