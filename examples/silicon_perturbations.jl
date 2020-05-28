using DFTK
using LinearAlgebra

include("perturbations.jl")
include("perturbations_tests.jl")

### Setting the model
# Calculation parameters
kgrid = [4, 4, 4]       # k-Point grid
supercell = [1, 1, 1]   # Lattice supercell

# Setup silicon lattice
a = 10.263141334305942  # Silicon lattice constant in Bohr
lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [0.008 .+ ones(3)/8, -ones(3)/8]]

# Make a supercell if desired
pystruct = pymatgen_structure(lattice, atoms)
pystruct.make_supercell(supercell)
lattice = load_lattice(pystruct)
atoms = [Si => [s.frac_coords for s in pystruct.sites]]

# precize the number of electrons on build the model
Ne = 8
model = model_LDA(lattice, atoms; n_electrons=Ne)

################################# Calculations #################################

avg = true
tol = 5e-15

kcoords, ksymops = bzmesh_ir_wedge(kgrid, model.lattice, model.atoms)
for Ecut in 5:5:25
    test_perturbation_ratio(Ecut, 120, 4, false, "gamma")
end

kcoords = [[0.27204337462860106, 0.4735127814871176, 0.6306195069419347]]
for Ecut in 5:5:25
    test_perturbation_ratio(Ecut, 120, 4, true, "nogamma")
end



