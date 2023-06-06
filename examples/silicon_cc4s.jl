# Very basic setup, useful for testing
using DFTK

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms     = [Si, Si]
positions = [ones(3)/8, -ones(3)/8]

model = model_LDA(lattice, atoms, positions)
basis = PlaneWaveBasis(model; Ecut=10, kgrid=[1, 1, 1])
scfres = self_consistent_field(basis, tol=1e-8)

DFTK.export_cc4s("silicon_cc4s.hdf5", scfres)
