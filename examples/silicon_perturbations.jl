#  using DFTK
using LinearAlgebra
using TimerOutputs

include("perturbations.jl")
include("perturbations_tests.jl")

reset_timer!(DFTK.timer)

### Setting the model
# Calculation parameters
kgrid = [4, 4, 4]       # k-Point grid
supercell = [1, 1, 1]   # Lattice supercell

# Setup silicon lattice
a = 10.263141334305942  # Silicon lattice constant in Bohr
lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [0.006 .+ ones(3)/8, -ones(3)/8]]

# Make a supercell if desired
pystruct = pymatgen_structure(lattice, atoms)
pystruct.make_supercell(supercell)
lattice = load_lattice(pystruct)
atoms = [Si => [s.frac_coords for s in pystruct.sites]]

# precize the number of electrons on build the model
model = model_LDA(lattice, atoms)

################################# Calculations #################################

avg = true
tol = 1e-10

kcoords, ksymops = bzmesh_ir_wedge(kgrid, model.symops)
α_list = vcat(collect(1:0.5:3))

filename = "simple_perturbation.h5"
h5open(filename, "w") do file
    file["alpha"] = α_list
end
test_perturbation_ratio(10, 40, false)


#  filename  = "perturbation_tests.h5"
#  Ecut_list = collect(5:1:8)
#  h5open(filename, "w") do file
#      file["alpha"] = α_list
#      file["Ecut_list"] = Ecut_list
#  end
#  for Ecut in Ecut_list
#      test_perturbation_ratio(Ecut, 30, true)
#  end


display(DFTK.timer)
