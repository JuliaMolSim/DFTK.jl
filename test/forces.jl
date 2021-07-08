using DFTK
import DFTK: mpi_mean!
using Test
using Random
using MPI
include("testcases.jl")

@testset "Forces on semiconductor (using total energy)" begin
    function energy_forces(pos)
        Ecut = 5
        Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
        atoms = [Si => pos]
        model = model_DFT(silicon.lattice, atoms, :lda_xc_teter93)
        basis = PlaneWaveBasis(model, Ecut, kgrid=[2, 2, 2], kshift=[0, 0, 0])

        is_converged = DFTK.ScfConvergenceDensity(1e-10)
        scfres = self_consistent_field(basis; is_converged=is_converged)
        scfres.energies.total, compute_forces(scfres), compute_forces_cart(scfres)
    end

    # symmetrical positions, forces should be 0
    pos0  = [(ones(3)) / 8, -ones(3) / 8]
    _, F0 = energy_forces(pos0)
    @test norm(F0) < 1e-4

    pos1 = [([1.01, 1.02, 1.03]) / 8, -ones(3) / 8]  # displace a bit from equilibrium
    disp = rand(3)
    mpi_mean!(disp, MPI.COMM_WORLD)  # must be identical on all processes
    ε = 1e-7
    pos2 = [pos1[1] + ε * disp, pos1[2]]

    E1, F1, Fc1 = energy_forces(pos1)
    E2,  _,  _  = energy_forces(pos2)

    diff_findiff = -(E2 - E1) / ε
    diff_forces = dot(F1[1][1], disp)
    @test abs(diff_findiff - diff_forces) < 1e-4

    # Rough test against QE reference (using PZ functional)
    reference = [[[-0.005622025, -0.00445816, -0.003278985],
                  [ 0.005622025,  0.00445816,  0.003278985]]]
    @test maximum(v -> maximum(abs, v), reference[1] - Fc1[1]) < 1e-5
end

@testset "Forces on metal (using free energy)" begin
    function silicon_energy_forces(pos; smearing=Smearing.FermiDirac())
        Ecut = 4
        Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
        atoms = [Si => pos]
        model = model_DFT(silicon.lattice, atoms, :lda_xc_teter93;
                          temperature=0.03, smearing=smearing, spin_polarization=:collinear)
        basis = PlaneWaveBasis(model, Ecut, kgrid=[4, 1, 2], kshift=[1/2, 0, 0])

        n_bands = 10
        is_converged = DFTK.ScfConvergenceDensity(5e-10)
        scfres = self_consistent_field(basis, n_bands=n_bands,
                                       is_converged=is_converged)
        scfres.energies.total, compute_forces(scfres)
    end

    pos1 = [([1.01, 1.02, 1.03]) / 8, -ones(3) / 8] # displace a bit from equilibrium
    disp = rand(3)
    mpi_mean!(disp, MPI.COMM_WORLD)  # must be identical on all processes
    ε = 1e-6
    pos2 = [pos1[1] + ε * disp, pos1[2]]

    for (tol, smearing) in [(0.003, Smearing.FermiDirac), (5e-5, Smearing.Gaussian)]
        E1, F1 = silicon_energy_forces(pos1; smearing=smearing())
        E2, _ = silicon_energy_forces(pos2; smearing=smearing())

        diff_findiff = -(E2 - E1) / ε
        diff_forces = dot(F1[1][1], disp)

        @test abs(diff_findiff - diff_forces) < tol
    end
end


@testset "Forces on spin-polarised case" begin
    function oxygen_energy_forces(pos)
        Ecut = 4
        O = ElementPsp(o2molecule.atnum, psp=load_psp("hgh/pbe/O-q6.hgh"))
        magnetic_moments = [O => [1.0, 1.0]]
        model = model_PBE(diagm([7.0, 7.0, 7.0]), [O => pos],
                          temperature=0.02, smearing=smearing=Smearing.Gaussian(),
                          magnetic_moments=magnetic_moments)
        basis = PlaneWaveBasis(model, Ecut; kgrid=[1, 1, 1])

        scfres = self_consistent_field(basis;
                                       is_converged=DFTK.ScfConvergenceDensity(1e-7),
                                       ρ=guess_density(basis, magnetic_moments),
                                       damping=0.7)
        scfres.energies.total, compute_forces(scfres)
    end

    pos1 = [[0, 0, 0.1155], [0.01, -2e-3, -0.2]]
    disp = rand(3)
    mpi_mean!(disp, MPI.COMM_WORLD)  # must be identical on all processes
    ε = 1e-6
    pos2 = [pos1[1] + ε * disp, pos1[2]]

    E1, F1 = oxygen_energy_forces(pos1)
    E2, _  = oxygen_energy_forces(pos2)

    diff_findiff = -(E2 - E1) / ε
    diff_forces  = dot(F1[1][1], disp)

    @test abs(diff_findiff - diff_forces) < 1e-4
end
