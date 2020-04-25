using DFTK
using Test
import DFTK: scf_determine_diagtol, scf_convergence_density_difference
include("testcases.jl")

@testset "Forces on semiconductor (using total energy)" begin
    function energy(pos)
        Ecut = 5
        Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
        atoms = [Si => pos]
        model = model_DFT(silicon.lattice, atoms, :lda_xc_teter93)
        basis = PlaneWaveBasis(model, Ecut, kgrid=[2, 1, 2])

        is_converged = scf_convergence_density_difference(1e-10)
        scfres = self_consistent_field(basis; is_converged=is_converged)
        sum(values(scfres.energies)), forces(scfres)
    end

    pos1 = [([1.01, 1.02, 1.03]) / 8, -ones(3) / 8]  # displace a bit from equilibrium
    disp = rand(3)
    ε = 1e-8
    pos2 = [pos1[1] + ε * disp, pos1[2]]

    E1, F1 = energy(pos1)
    E2, _ = energy(pos2)

    diff_findiff = -(E2 - E1) / ε
    diff_forces = dot(F1[1][1], disp)
    @test abs(diff_findiff - diff_forces) < 2e-6
end

@testset "Forces on metal (using free energy)" begin
    function silicon_energy_forces(pos; smearing=Smearing.FermiDirac())
        Ecut = 4
        Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
        atoms = [Si => pos]
        model = model_DFT(silicon.lattice, atoms, :lda_xc_teter93;
                          temperature=0.03, smearing=smearing)
        basis = PlaneWaveBasis(model, Ecut, kgrid=[2, 1, 2])

        n_bands = 10
        is_converged = scf_convergence_density_difference(5e-11)
        determine_diagtol = scf_determine_diagtol(ratio_ρdiff=0.01)
        scfres = self_consistent_field(basis, n_bands=n_bands,
                                       is_converged=is_converged,
                                       determine_diagtol=determine_diagtol
                                      )
        sum(values(scfres.energies)), forces(scfres)
    end

    pos1 = [([1.01, 1.02, 1.03]) / 8, -ones(3) / 8] # displace a bit from equilibrium
    disp = rand(3)
    ε = 1e-8
    pos2 = [pos1[1] + ε * disp, pos1[2]]

    for smearing in [Smearing.FermiDirac, Smearing.Gaussian]
        E1, F1 = silicon_energy_forces(pos1; smearing=smearing())
        E2, _ = silicon_energy_forces(pos2; smearing=smearing())

        diff_findiff = -(E2 - E1) / ε
        diff_forces = dot(F1[1][1], disp)

        @test abs(diff_findiff - diff_forces) < 2e-5
    end
end
