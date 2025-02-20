using AtomsBuilder
using DFTK
using CUDA
using PseudoPotentialData

model  = model_DFT(bulk(:Si);
                   functionals=PBE(),
                   pseudopotentials=PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf"))

# If available, use CUDA to store DFT quantities and perform main computations
architecture = has_cuda() ? DFTK.GPU(CuArray) : DFTK.CPU()

basis  = PlaneWaveBasis(model; Ecut=30, kgrid=(5, 5, 5), architecture)
scfres = self_consistent_field(basis; tol=1e-6)
