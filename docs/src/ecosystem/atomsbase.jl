# # AtomsBase integration

# [AtomsBase.jl](https://github.com/JuliaMolSim/AtomsBase.jl) is a common interface
# for representing atomic structures in Julia. DFTK directly supports using such
# structures to run a calculation as is demonstrated here.

using DFTK
using AtomsBuilder

# ## Feeding an AtomsBase AbstractSystem to DFTK
#
# In this example we construct a bulk silicon system using the `bulk` function
# from [AtomsBuilder](https://github.com/JuliaMolSim/AtomsBuilder.jl). This function
# uses tabulated data to set up a reasonable starting geometry and lattice for bulk silicon.

system = bulk(:Si)

# By default the atoms of an `AbstractSystem` employ the bare Coulomb potential.
# To employ pseudpotential models (which is almost always advisable for
# plane-wave DFT) one employs the `pseudopotential` keyword argument in
# model constructors such as [`model_DFT`](@ref).
# For example we can employ a `PseudoFamily` object
# from the [PseudoPotentialData](https://github.com/JuliaMolSim/PseudoPotentialData.jl)
# package. See its documentation for more information on the available
# pseudopotential families and how to select them.

using PseudoPotentialData  # defines PseudoFamily

pd_lda_family = PseudoFamily("dojo.nc.sr.lda.v0_4_1.oncvpsp3.standard.upf")
model = model_DFT(system;
                  functionals=LDA(),
                  temperature=1e-3,
                  pseudopotentials=pd_lda_family)

# Alternatively the `pseudopotentials` object also accepts a `Dict{Symbol,String}`,
# which provides for each element symbol the filename or identifier
# of the pseudopotential to be employed, e.g.

model = model_DFT(system;
                  functionals=LDA(),
                  temperature=1e-3,
                  pseudopotentials=Dict(:Si => "hgh/lda/si-q4"))

# We can then discretise such a model and solve:
basis  = PlaneWaveBasis(model; Ecut=15, kgrid=[4, 4, 4])
scfres = self_consistent_field(basis, tol=1e-8);

# If we did not want to use
# [AtomsBuilder](https://github.com/JuliaMolSim/AtomsBuilder.jl)
# we could of course use any other package
# which yields an AbstractSystem object. This includes:

# ### Reading a system using AtomsIO
#
using AtomsIO

## Read a file using [AtomsIO](https://github.com/mfherbst/AtomsIO.jl),
## which directly yields an AbstractSystem.
system = load_system("Si.extxyz")

## Now run the LDA calculation:
pseudopotentials = Dict(:Si => "hgh/lda/si-q4")
model  = model_DFT(system; pseudopotentials, functionals=LDA(), temperature=1e-3)
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
pseudopotentials = Dict(:Si => "hgh/lda/si-q4")
model  = model_DFT(system; pseudopotentials, functionals=LDA(), temperature=1e-3)
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
Si = ElementPsp(:Si, load_psp("hgh/lda/Si-q4"))
atoms     = [Si, Si]
positions = [ones(3)/8, -ones(3)/8]

third_system = atomic_system(lattice, atoms, positions)
