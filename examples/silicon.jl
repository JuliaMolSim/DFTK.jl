using DFTK
using PyPlot

# Calculation parameters
kgrid = [4, 4, 4]       # k-Point grid
supercell = [1, 1, 1]   # Lattice supercell
Ecut = 30               # kinetic energy cutoff in Hartree
n_bands = 8             # number of bands to plot in the bandstructure

# Setup silicon lattice
a = 10.263141334305942  # Silicon lattice constant in Bohr
lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

# Make a supercell if desired
pystruct = pymatgen_structure(lattice, atoms)
pystruct.make_supercell(supercell)
lattice = load_lattice(pystruct)
atoms = [Si => [s.frac_coords for s in pystruct.sites]]

model = model_LDA(lattice, atoms)
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid, enable_bzmesh_symmetry=false)

# Run SCF. Note Silicon is a semiconductor, so we use an insulator
# occupation scheme. This will cause warnings in some models, because
# e.g. in the :reduced_hf model silicon is a metal
# scfres = direct_minimization(basis)
scfres = self_consistent_field(basis, tol=1e-10)

# compute forces
#  F = forces(scfres)
#  println(F)

# Print energies and plot bands
println()
display(scfres.energies)
#  gui(plot_bandstructure(scfres, n_bands))
