# # Creating supercells with pymatgen
#
# The [Pymatgen](https://pymatgen.org/) python library allows to setup
# solid-state calculations using a flexible set of classes as well as an API
# to an online data base of structures. Its `Structure` and `Lattice`
# objects are directly supported by the DFTK `load_atoms` and `load_lattice`
# functions, such that DFTK may be readily used to run calculation on systems
# defined in pymatgen. Using the `pymatgen_structure` function a conversion
# from DFTK to pymatgen structures is also possible. In the following we
# use this to create a silicon supercell and find its LDA ground state
# using direct minimisation. To run this example Julia's `PyCall` package
# needs to be able to find an installation of `pymatgen`.

# First we setup the silicon lattice in DFTK.
using DFTK

a = 10.263141334305942  # Lattice constant in Bohr
lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]];

# Next we make a `[2, 2, 2]` supercell using pymatgen
pystruct = pymatgen_structure(lattice, atoms)
pystruct.make_supercell([2, 2, 2])
lattice = load_lattice(pystruct)
atoms = [Si => [s.frac_coords for s in pystruct.sites]];

# Setup an LDA model and discretize using
# a single k-point and a small `Ecut` of 5 Hartree.
model = model_LDA(lattice, atoms)
basis = PlaneWaveBasis(model; Ecut=5, kgrid=(1, 1, 1))

# Find the ground state using direct minimisation (always using SCF is boring ...)
scfres = direct_minimization(basis, tol=1e-5);
#-
scfres.energies
