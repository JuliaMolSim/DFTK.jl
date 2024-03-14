# # AtomsBase integration

# [AtomsBase.jl](https://github.com/JuliaMolSim/AtomsBase.jl) is a common interface
# for representing atomic structures in Julia. DFTK directly supports using such
# structures to run a calculation as is demonstrated here.

using DFTK

# ## Feeding an AtomsBase AbstractSystem to DFTK
# In this example we construct a silicon system using the `ase.build.bulk` routine
# from the [atomistic simulation environment](https://wiki.fysik.dtu.dk/ase/index.html)
# (ASE), which is exposed by [ASEconvert](https://github.com/mfherbst/ASEconvert.jl)
# as an AtomsBase `AbstractSystem`.

## Construct bulk system and convert to an AbstractSystem
using ASEconvert
system_ase = ase.build.bulk("Si")
system = pyconvert(AbstractSystem, system_ase)

# To use an AbstractSystem in DFTK, we attach pseudopotentials, construct a DFT model,
# discretise and solve:
system = attach_psp(system; Si="hgh/lda/si-q4")

model  = model_LDA(system; temperature=1e-3)
basis  = PlaneWaveBasis(model; Ecut=15, kgrid=[4, 4, 4])
scfres = self_consistent_field(basis, tol=1e-8);

# If we did not want to use ASE we could of course use any other package
# which yields an AbstractSystem object. This includes:

# ### Reading a system using AtomsIO
#
using AtomsIO

## Read a file using [AtomsIO](https://github.com/mfherbst/AtomsIO.jl),
## which directly yields an AbstractSystem.
system = load_system("Si.extxyz")

## Now run the LDA calculation:
system = attach_psp(system; Si="hgh/lda/si-q4")
model  = model_LDA(system; temperature=1e-3)
basis  = PlaneWaveBasis(model; Ecut=15, kgrid=[4, 4, 4])
scfres = self_consistent_field(basis, tol=1e-8);

# The same could be achieved using [ExtXYZ](https://github.com/libAtoms/ExtXYZ.jl)
# by `system = Atoms(read_frame("Si.extxyz"))`,
# since the `ExtXYZ.Atoms` object is directly AtomsBase-compatible.

# ### Directly setting up a system in AtomsBase

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

## Now run the LDA calculation:
system = attach_psp(system; Si="hgh/lda/si-q4")
model  = model_LDA(system; temperature=1e-3)
basis  = PlaneWaveBasis(model; Ecut=15, kgrid=[4, 4, 4])
scfres = self_consistent_field(basis, tol=1e-4);

# ## Obtaining an AbstractSystem from DFTK data

# At any point we can also get back the DFTK model as an
# AtomsBase-compatible `AbstractSystem`:
second_system = atomic_system(model)

# Similarly DFTK offers a method to the `atomic_system` and `periodic_system` functions
# (from AtomsBase), which enable a seamless conversion of the usual data structures for
# setting up DFTK calculations into an `AbstractSystem`:
lattice = 5.431u"Å" / 2 * [[0 1 1.];
                           [1 0 1.];
                           [1 1 0.]];
Si = ElementPsp(:Si; psp=load_psp("hgh/lda/Si-q4"))
atoms     = [Si, Si]
positions = [ones(3)/8, -ones(3)/8]

third_system = atomic_system(lattice, atoms, positions)
