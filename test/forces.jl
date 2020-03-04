using DFTK
using Test
include("testcases.jl")

@testset "Forces on semiconductor (using total energy)" begin
    function energy(pos)
        Ecut = 5
        Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
        atoms = [Si => pos]
        model = model_dft(silicon.lattice, :lda_xc_teter93, atoms)
        basis = PlaneWaveBasis(model, Ecut, kgrid=[2, 1, 2])

        n_bands = 4
        ham = Hamiltonian(basis, guess_density(basis))
        scfres = self_consistent_field(ham, n_bands, tol=1e-10, diagtol=1e-10)
        sum(values(scfres.energies)), forces(scfres)
    end

    pos1 = [(ones(3) + 0.1randn(3)) / 8, -ones(3) / 8]
    disp = randn(3)
    ε = 1e-8
    pos2 = [pos1[1] + ε * disp, pos1[2]]

    E1, F1 = energy(pos1)
    E2, _ = energy(pos2)

    diff_findiff = -(E2 - E1) / ε
    diff_forces = dot(F1[1][1], disp)
    @test abs(diff_findiff - diff_forces) < 1.5e-6
end


@testset "Forces on metal (using free energy)" begin
    function silicon_energy_forces(pos; ρ=nothing, smearing=Smearing.FermiDirac())
        Ecut = 5
        Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
        atoms = [Si => pos]
        model = model_dft(silicon.lattice, :lda_xc_teter93, atoms,
                          temperature=0.03, smearing=smearing)
        basis = PlaneWaveBasis(model, Ecut, kgrid=[2, 1, 2])

        n_bands = 10
        ρguess = guess_density(basis)
        if !isnothing(ρ)
            α = 1e-4
            ρguess = from_real(basis, (1-α) * ρ.real + α * ρguess.real)
        end
        ham = Hamiltonian(basis, ρguess)
        scfres = self_consistent_field(ham, n_bands, tol=1e-12, diagtol=1e-12)
        sum(values(scfres.energies)), forces(scfres), scfres.ρ
    end

    pos1 = [(ones(3) + 0.1randn(3)) / 8, -ones(3) / 8]
    disp = randn(3)
    ε = 1e-8
    pos2 = [pos1[1] + ε * disp, pos1[2]]

    for smearing in [Smearing.FermiDirac, Smearing.Gaussian]
        E1, F1, ρ = silicon_energy_forces(pos1; smearing=smearing())
        E2, _, _ = silicon_energy_forces(pos2; ρ=ρ, smearing=smearing())

        diff_findiff = -(E2 - E1) / ε
        diff_forces = dot(F1[1][1], disp)

        @test abs(diff_findiff - diff_forces) < 1e-5
    end
end
