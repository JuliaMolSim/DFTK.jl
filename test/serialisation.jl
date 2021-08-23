using DFTK
using JLD2
using WriteVTK
using MPI
import DFTK: ScfDefaultCallback, ScfSaveCheckpoints
using Test
include("testcases.jl")

function test_scfres_agreement(tested, ref)
    @test tested.basis.model.lattice           == ref.basis.model.lattice
    @test tested.basis.model.smearing          == ref.basis.model.smearing
    @test tested.basis.model.symmetries        == ref.basis.model.symmetries
    @test tested.basis.model.spin_polarization == ref.basis.model.spin_polarization

    @test length(tested.basis.model.atoms) == length(ref.basis.model.atoms)
    @test tested.basis.model.atoms[1][2]   == ref.basis.model.atoms[1][2]

    @test tested.basis.Ecut      == ref.basis.Ecut
    @test tested.basis.kweights  == ref.basis.kweights
    @test tested.basis.ksymops   == ref.basis.ksymops
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


@testset "Test checkpointing" begin
    O = ElementPsp(o2molecule.atnum, psp=load_psp("hgh/pbe/O-q6.hgh"))
    magnetic_moments = [O => [1., 1.]]
    model = model_PBE(o2molecule.lattice, [O => o2molecule.positions],
                      temperature=0.02, smearing=smearing=Smearing.Gaussian(),
                      magnetic_moments=magnetic_moments, symmetries=false)

    kgrid = [1, mpi_nprocs(), 1]   # Ensure at least 1 kpt per process
    basis  = PlaneWaveBasis(model; Ecut=4, kgrid=kgrid)

    # Run SCF and do checkpointing along the way
    mktempdir() do tmpdir
        checkpointfile = joinpath(tmpdir, "scfres.jld2")
        checkpointfile = MPI.bcast(checkpointfile, 0, MPI.COMM_WORLD)  # master -> everyone

        callback = ScfDefaultCallback() ∘ ScfSaveCheckpoints(checkpointfile; keep=true)
        scfres = self_consistent_field(basis, tol=5e-2, callback=callback)
        test_scfres_agreement(scfres, load_scfres(checkpointfile))
    end
end

@testset "Test serialisation" begin
    Si = ElementPsp(14, psp=load_psp(silicon.psp))
    atoms = [Si => silicon.positions]
    model = model_LDA(silicon.lattice, atoms, spin_polarization=:collinear, temperature=0.01)
    kgrid = [2, 3, 4]
    basis = PlaneWaveBasis(model; Ecut=5, kgrid=kgrid)
    scfres = self_consistent_field(basis)

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
