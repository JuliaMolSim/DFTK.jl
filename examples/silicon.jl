# Very basic setup, useful for testing
using DFTK
using PseudoPotentialData

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth"))
atoms     = [Si, Si]
positions = [ones(3)/8, -ones(3)/8]

model = model_DFT(lattice, atoms, positions; functionals=LDA())
basis = PlaneWaveBasis(model; Ecut=15, kgrid=[4, 4, 4])
scfres = self_consistent_field(basis, tol=1e-8)
