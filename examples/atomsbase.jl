using DFTK
using AtomsBase
using Unitful
using UnitfulAtomic

# Construct system in the AtomsBase world
a = 10.26u"bohr"  # Silicon lattice constant
lattice = a / 2 * [[0, 1, 1.],  # Lattice as vector of vectors
                   [1, 0, 1.],
                   [1, 1, 0.]]
atoms  = [:Si => ones(3)/8, :Si => -ones(3)/8]
system = periodic_system(atoms, lattice; fractional=true)

# Use it inside DFTK:
# Attach pseudopotentials, construct appropriate model,
# discretise and solve.
system = attach_psp(system; family="hgh", functional="lda")

model  = model_LDA(system; temperature=1e-3)
basis  = PlaneWaveBasis(model; Ecut=15, kgrid=[4, 4, 4])
scfres = self_consistent_field(basis, tol=1e-8)

# Get the DFTK model back into the AtomsBase world:
newsystem = atomic_system(model)
@show atomic_number(newsystem)
