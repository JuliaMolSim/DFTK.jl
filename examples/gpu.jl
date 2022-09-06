using DFTK
using CUDA

a = 10.263141334305942  # Lattice constant in Bohr
lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms     = [Si, Si]
positions = [ones(3)/8, -ones(3)/8]
terms = [Kinetic(),
            AtomicLocal(),
            AtomicNonlocal(),
            Ewald(),
            PspCorrection(),
            Entropy(),
            Hartree()]
# Now, build a supercell to have a larger system
pystruct = pymatgen_structure(lattice, atoms, positions)
pystruct.make_supercell([4,2,2])
lattice   = load_lattice(pystruct)
positions = load_positions(pystruct)
atoms     = fill(Si, length(positions))

model = Model(lattice, atoms, positions; terms=terms, temperature=1e-3, symmetries=false)
# Notice the only difference in the code, with the optional argument array_type
basis_gpu = PlaneWaveBasis(model; Ecut=30, kgrid=(1, 1, 1), array_type = CuArray)
# You can now check that some of the fields of the basis, such as the G_vectors, are CuArrays

scfres = self_consistent_field(basis_gpu; tol=1e-3, solver=scf_anderson_solver(), mixing = KerkerMixing())
