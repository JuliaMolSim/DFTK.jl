using DFTK
using Test
function energy(pos)
    # Calculation parameters
    kgrid = [2, 1, 2]        # k-Point grid
    supercell = [1, 1, 1]    # Lattice supercell
    Ecut = 5                # kinetic energy cutoff in Hartree

    # Setup silicon lattice
    a = 10.263141334305942  # Silicon lattice constant in Bohr
    lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
    Si = AtomType(14, psp=load_psp("hgh/lda/Si-q4"))
    # Si = AtomType(14)
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
