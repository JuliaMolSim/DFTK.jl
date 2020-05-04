using DFTK
using PyPlot
import Statistics: mean

include("perturbations.jl")

### Setting the model
# Calculation parameters
kgrid = [1, 1, 1]       # k-Point grid
supercell = [1, 1, 1]   # Lattice supercell

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

# precize the number of electrons on build the model
Ne = 8
model = model_LDA(lattice, atoms; n_electrons=Ne)

# kgrid and ksymops
kcoords, ksymops = bzmesh_ir_wedge(kgrid, model.lattice, model.atoms)
# modify kcoords to use different k points
#  kcoords = [[rand(), rand(), rand()]]
#  a specific kcoords for which we know it works
#  kcoords = [[0.27204337462860106, 0.4735127814871176, 0.6306195069419347]]
println(kcoords)

################################# Calculations #################################

avg = true
#  test_perturbation_ratio(15, 100, 3)
test_perturbation_coarsegrid(2.5, 5, 80)


