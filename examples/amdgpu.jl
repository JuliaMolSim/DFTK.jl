using AtomsBuilder
using DFTK
using AMDGPU
using PseudoPotentialData

model  = model_DFT(bulk(:Si);
                   functionals=PBE(),
                   pseudopotentials=PseudoFamily("cp2k.nc.sr.pbe.v0_1.semicore.gth"))

# If available, use AMDGPU to store DFT quantities and perform main computations
architecture = has_rocm_gpu() ? DFTK.GPU(AMDGPU.ROCArray) : DFTK.CPU()

basis  = PlaneWaveBasis(model; Ecut=20, kgrid=(1, 1, 1), architecture)

# Anderson does not yet work ...
scfres = self_consistent_field(basis; tol=1e-2, solver=scf_damping_solver())
