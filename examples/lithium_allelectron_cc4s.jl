# Very basic setup, useful for testing
using DFTK
using ASEconvert
using AtomsBase

system = ase.build.bulk("Li")
model = model_LDA(pyconvert(AbstractSystem, system); temperature=1e-3, smearing=Smearing.Gaussian())
basis = PlaneWaveBasis(model; Ecut=10, kgrid=[1, 1, 1])
scfres = self_consistent_field(basis, tol=1e-8)

DFTK.export_cc4s("silicon_cc4s.hdf5", scfres)
