# Very basic setup, useful for testing
using DFTK
using ASEconvert
using AtomsBase

system = ase.build.bulk("Li")
model  = model_LDA(pyconvert(AbstractSystem, system); temperature=1e-3, smearing=Smearing.Gaussian())
basis  = PlaneWaveBasis(model; Ecut=15, kgrid=[1, 1, 1])
scfres = DFTK.scf_potential_mixing_adaptive(basis, tol=1e-8)

bands = compute_bands(scfres, basis.kgrid; n_bands=30, tol=1e-10)
DFTK.export_cc4s(bands, joinpath(pwd(), "cc4s_lithium"); force=true)
