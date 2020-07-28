#  using DFTK
using LinearAlgebra
using TimerOutputs

include("perturbations.jl")
include("perturbations_tests.jl")

reset_timer!(DFTK.timer)

### Setting the model
a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]
tol = 1e-12
nel = 8

model = model_LDA(lattice, atoms, n_electrons=nel)
kgrid = [3, 3, 3]  # k-point grid (Regular Monkhorst-Pack grid)
Ecut_ref = 60           # kinetic energy cutoff in Hartree
kcoords, ksymops = bzmesh_ir_wedge(kgrid, model.symops)

################################# Calculations #################################

avg = true
tol = 1e-10

#  kcoords, ksymops = bzmesh_ir_wedge(kgrid, model.symops)
#  α_list = vcat(collect(1:0.5:3))

#  filename = "simple_perturbation.h5"
#  h5open(filename, "w") do file
#      file["alpha"] = α_list
#  end
#  test_perturbation_ratio(10, 40, false)


#  #  filename  = "perturbation_tests.h5"
#  #  Ecut_list = collect(5:1:8)
#  #  h5open(filename, "w") do file
#  #      file["alpha"] = α_list
#  #      file["Ecut_list"] = Ecut_list
#  #  end
#  #  for Ecut in Ecut_list
#  #      test_perturbation_ratio(Ecut, 30, true)
#  #  end

filename = "improvement_ratio.h5"
res = test_perturbation_coarsegrid(2.5, 4, 76)


display(DFTK.timer)
