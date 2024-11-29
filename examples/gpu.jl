using AtomsBuilder
using DFTK
using CUDA

system = attach_psp(bulk(:Si); Si="hgh/pbe/Si-q4")
model  = model_DFT(system; functionals=PBE())

# If available, use CUDA to store DFT quantities and perform main computations
architecture = has_cuda() ? DFTK.GPU(CuArray) : DFTK.CPU()

basis  = PlaneWaveBasis(model; Ecut=30, kgrid=(5, 5, 5), architecture)
scfres = self_consistent_field(basis; tol=1e-2, solver=scf_damping_solver())
