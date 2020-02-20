using DFTK
using Test
include("testcases.jl")

@testset "Forces on semiconductor (using total energy)" begin
    function energy(pos)
        Ecut = 5                # kinetic energy cutoff in Hartree

        # Setup silicon lattice
        lattice = silicon.lattice
        Si = Element(silicon.atnum, psp=load_psp(silicon.psp))
        atoms = [Si => pos]
        model = model_dft(silicon.lattice, :lda_xc_teter93, atoms)
        basis = PlaneWaveBasis(model, Ecut, kgrid=[2, 1, 2])

        n_bands_scf = Int(model.n_electrons / 2)
        ham = Hamiltonian(basis, guess_density(basis))
        scfres = self_consistent_field(ham, n_bands_scf, tol=1e-12)

        sum(values(scfres.energies)), forces(scfres)
    end

    pos1 = [(ones(3)+0.1*randn(3))/8, -ones(3)/8]
    disp = randn(3)
    ε = 1e-8
    pos2 = [pos1[1]+ε*disp, pos1[2]]

    E1, F1 = energy(pos1)
    E2, F2 = energy(pos2)

    diff_findiff = -(E2-E1)/ε
    diff_forces = dot(F1[1][1], disp)

    @test abs(diff_findiff - diff_forces) < 1e-6
end
