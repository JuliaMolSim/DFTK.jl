#  using DFTK
using LinearAlgebra
using FFTW
using TimerOutputs

BLAS.set_num_threads(4)
FFTW.set_num_threads(4)

include("perturbations.jl")
include("perturbations_tests.jl")

reset_timer!(DFTK.timer)

### Setting the model
a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [.008 .+ ones(3)/8, -ones(3)/8]]
tol = 1e-10
nel = 8

model = model_LDA(lattice, atoms, n_electrons=nel)
kgrid = [3, 3, 3]  # k-point grid (Regular Monkhorst-Pack grid)
kcoords, ksymops = bzmesh_ir_wedge(kgrid, model.symops)

################################# Calculations #################################

α_list = vcat(collect(1:0.2:3))

filename  = "perturbation_tests.h5"
Ecut_list = collect(5:5:10)
Ecut_ref = 40

h5open(filename, "w") do file
    file["alpha"] = α_list
    file["Ecut_list"] = Ecut_list
    file["Ecut_ref"] = Ecut_ref
    file["nk"] = length(kcoords)
end
for Ecut in Ecut_list
    test_perturbation_ratio(Ecut, Ecut_ref, true)
end

#  filename = "improvement_ratio.h5"
#  res = test_perturbation_coarsegrid(2.5, 4, 76)

display(DFTK.timer)
