using DFTK
using Test
include("testcases.jl")

@testset "Forces on semiconductor (using total energy)" begin
    function energy(pos)
        Ecut = 5
        Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
        atoms = [Si => pos]
        model = model_DFT(silicon.lattice, atoms, :lda_xc_teter93)
        basis = PlaneWaveBasis(model, Ecut, kgrid=[2, 2, 2])

        is_converged = DFTK.ScfConvergenceDensity(1e-10)
        scfres = self_consistent_field(basis; is_converged=is_converged)
        scfres.energies.total, forces(scfres)
    end

    # symmetrical positions, forces should be 0
    pos0 = [(ones(3)) / 8, -ones(3) / 8]
    _, F1 = energy(pos0)
    @test norm(F1) < 1e-4

    pos1 = [([1.01, 1.02, 1.03]) / 8, -ones(3) / 8]  # displace a bit from equilibrium
    disp = rand(3)
    ε = 1e-8
    pos2 = [pos1[1] + ε * disp, pos1[2]]

    E1, F1 = energy(pos1)
    E2, _ = energy(pos2)

    diff_findiff = -(E2 - E1) / ε
    diff_forces = dot(F1[1][1], disp)
    @test abs(diff_findiff - diff_forces) < 1e-4
end

@testset "Forces on metal (using free energy)" begin
    function silicon_energy_forces(pos; smearing=Smearing.FermiDirac())
        Ecut = 4
        Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
        atoms = [Si => pos]
        model = model_DFT(silicon.lattice, atoms, :lda_xc_teter93;
                          temperature=0.03, smearing=smearing)
        # TODO Put kshift=[1/2, 0, 0] here later
        basis = PlaneWaveBasis(model, Ecut, kgrid=[2, 1, 2])

        n_bands = 10
        is_converged = DFTK.ScfConvergenceDensity(5e-10)
        scfres = self_consistent_field(basis, n_bands=n_bands,
                                       is_converged=is_converged,
                                      )
        scfres.energies.total, forces(scfres)
    end

    pos1 = [([1.01, 1.02, 1.03]) / 8, -ones(3) / 8] # displace a bit from equilibrium
    disp = rand(3)
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
