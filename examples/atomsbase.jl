# # AtomsBase integration

# [AtomsBase.jl](https://github.com/JuliaMolSim/AtomsBase.jl) is a common interface
# for representing atomic structures in Julia. DFTK directly supports using such
# structures to run a calculation as is demonstrated here.

using DFTK
using AtomsBase
using Unitful
using UnitfulAtomic

## Construct a system in the AtomsBase world
a = 10.26u"bohr"  # Silicon lattice constant
lattice = a / 2 * [[0, 1, 1.],  # Lattice as vector of vectors
                   [1, 0, 1.],
                   [1, 1, 0.]]
atoms  = [:Si => ones(3)/8, :Si => -ones(3)/8]
system = periodic_system(atoms, lattice; fractional=true)

# System is an AtomsBase-compatible system. To use it in DFTK,
# we attach pseudopotentials, construct a DFT model, discretise and solve:
system = attach_psp(system; family="hgh", functional="lda")

model  = model_LDA(system; temperature=1e-3)
basis  = PlaneWaveBasis(model; Ecut=15, kgrid=[4, 4, 4])
scfres = self_consistent_field(basis, tol=1e-4);

# At any point we can also get back the DFTK model as an
# AtomsBase-compatible `AbstractSystem`:
newsystem = atomic_system(model)
