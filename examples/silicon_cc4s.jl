# Very basic setup, useful for testing
using DFTK

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms     = [Si, Si]
positions = [ones(3)/8, -ones(3)/8]

# TODO This should be Hartree-Fock ... otherwise the result is not proper CC !
model  = model_LDA(lattice, atoms, positions)
basis  = PlaneWaveBasis(model; Ecut=10, kgrid=[1, 1, 1])
scfres = self_consistent_field(basis, tol=1e-8)

bands = compute_bands(scfres, basis.kgrid; n_bands=10, tol=1e-10)
DFTK.export_cc4s(bands, joinpath(pwd(), "cc4s_silicon"); force=true)
