using AtomsBuilder
using DFTK
using CUDA

model  = model_DFT(bulk(:Si);
                   functionals=PBE(),
                   pseudopotentials=Dict(:Si => "hgh/pbe/Si-q4"))

# If available, use CUDA to store DFT quantities and perform main computations
architecture = has_cuda() ? DFTK.GPU(CuArray) : DFTK.CPU()

basis  = PlaneWaveBasis(model; Ecut=30, kgrid=(5, 5, 5), architecture)
scfres = self_consistent_field(basis; tol=1e-2, solver=scf_damping_solver())
