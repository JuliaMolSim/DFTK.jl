using DFTK
using JLD2
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

    @test tested.n_iter         == ref.n_iter
    @test tested.energies.total ≈  ref.energies.total atol=1e-13
    @test tested.eigenvalues    == ref.eigenvalues
    @test tested.occupation     == ref.occupation
    @test tested.ψ              == ref.ψ
    @test tested.ρ.real         == ref.ρ.real
    @test tested.ρspin.real     == ref.ρspin.real
end


if mpi_nprocs() == 1  # Not yet implemented for MPI-parallel runs
    @testset "Test checkpointing" begin
        O = ElementPsp(o2molecule.atnum, psp=load_psp("hgh/pbe/O-q6.hgh"))
        magnetic_moments = [O => [1., 1.]]
        model = model_PBE(o2molecule.lattice, [O => o2molecule.positions],
                          temperature=0.02, smearing=smearing=Smearing.Gaussian(),
                          magnetic_moments=magnetic_moments)
        basis  = PlaneWaveBasis(model, 4; kgrid=[1, 1, 1])
        ρspin  = guess_spin_density(basis, magnetic_moments)

        # Run SCF and do checkpointing along the way
        scfres = mktempdir() do tmpdir
            checkpointfile = joinpath(tmpdir, "scfres.jld2")
            callback = ScfDefaultCallback() ∘ ScfSaveCheckpoints(checkpointfile; keep=true)
            scfres = self_consistent_field(basis, tol=5e-2, ρspin=ρspin, callback=callback)
            test_scfres_agreement(scfres, load_scfres(checkpointfile))
            scfres
        end

        # Save & load scfres
        mktempdir() do tmpdir
            checkpointfile = joinpath(tmpdir, "scfres.jld2")
            save_scfres(checkpointfile, scfres)
            test_scfres_agreement(scfres, load_scfres(checkpointfile))
        end
    end
end
