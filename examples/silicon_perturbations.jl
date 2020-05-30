using DFTK
using LinearAlgebra

include("perturbations.jl")
include("perturbations_tests.jl")

### Setting the model
# Calculation parameters
kgrid = [1, 1, 1]       # k-Point grid
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
model = model_LDA(lattice, atoms)

################################# Calculations #################################

avg = true
tol = 1e-12

kcoords, ksymops = bzmesh_ir_wedge(kgrid, model.lattice, model.atoms)
Ecut_list = collect(5:1:8)
α_list = vcat(collect(1:0.5:3))

h5open("perturbation_tests.h5", "w") do file
    file["alpha"] = α_list
    file["nel"] = nel
    file["Ecut_list"] = Ecut_list
end
for Ecut in Ecut_list
    test_perturbation_ratio(Ecut, 30, 3, true)
end

#  kcoords = [[0.27204337462860106, 0.4735127814871176, 0.6306195069419347]]
#  for Ecut in 5:5:25
#      test_perturbation_ratio(Ecut, 120, 4, true)
#  end



