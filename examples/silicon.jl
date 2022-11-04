# Very basic setup, useful for testing
using DFTK
using Unitful
using UnitfulAtomic

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms     = [Si, Si]
positions = [ones(3)/8, -ones(3)/8]

# model = model_LDA(lattice, atoms, positions, temperature=1e-2, smearing=Smearing.MethfesselPaxton(2))
model = model_LDA(lattice, atoms, positions, temperature=1e-2u"Ry", smearing=Smearing.MarzariVanderbilt())
basis = PlaneWaveBasis(model; Ecut=15, kgrid=[4, 4, 4])
scfres = self_consistent_field(basis, tol=1e-8, mixing=LdosMixing())
nothing
