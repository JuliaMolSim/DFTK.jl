using DFTK
using Test

@testset "Forces" begin
    include("testcases.jl")
    function energy(pos)
        kgrid = [2, 1, 2]        # k-Point grid
        Ecut = 5                # kinetic energy cutoff in Hartree

        # Setup silicon lattice
        lattice = silicon.lattice
        Si = Element(silicon.atnum, psp=load_psp(silicon.psp))
        atoms = [Si => pos]

        model = Model(lattice;
                      atoms=atoms,
                      external=term_external(atoms),
                      nonlocal=term_nonlocal(atoms),
                      hartree=term_hartree(),
                      xc=term_xc(:lda_xc_teter93))

        kcoords, ksymops = bzmesh_ir_wedge(kgrid, lattice, atoms)
        basis = PlaneWaveBasis(model, Ecut, kcoords, ksymops)

        # Run SCF. Note Silicon is a semiconductor, so we use an insulator
        # occupation scheme. This will cause warnings in some models, because
        # e.g. in the :reduced_hf model silicon is a metal
        n_bands_scf = Int(model.n_electrons / 2)
        ham = Hamiltonian(basis, guess_density(basis))
        scfres = self_consistent_field(ham, n_bands_scf, tol=1e-12)
        ham = scfres.ham

        energies = scfres.energies
        sum(values(energies)), forces(scfres)
    end

    using Random
    Random.seed!(0)

    pos1 = [(ones(3)+.1*randn(3))/8, -ones(3)/8]
    disp = randn(3)
    ε = 1e-7
    pos2 = [pos1[1]+ε*disp, pos1[2]]

    E1, F1 = energy(pos1)
    E2, F2 = energy(pos2)

    diff_findiff = -(E2-E1)/ε
    diff_forces = dot(F1[1][1], disp)

    @test abs(diff_findiff - diff_forces) < 1e-6
end
