using DFTK
using PyCall
using Plots

slab = pyimport("ase.build").fcc100("Al", (2, 2, 2), a=4.05, vacuum=7.5)
pyimport("ase.build").add_adsorbate(slab, "Na", 4.0)
slab.center(axis=2)
slab.pbc = true

lattice = load_lattice(slab)
atoms   = load_atoms(slab)
atoms   = [ElementPsp(el.symbol, psp=load_psp(el.symbol, functional="lda")) => position
           for (el, position) in atoms]

model  = model_LDA(lattice, atoms, extra_terms=[DFTK.SurfaceDipoleCorrection()],
                   temperature=0.01, smearing=Smearing.Gaussian())
# basis  = PlaneWaveBasis(model, 15, kgrid=[4, 4, 1])
basis  = PlaneWaveBasis(model, 15, kgrid=[1, 1, 1])
scfres = self_consistent_field(basis, tol=1e-6, mixing=HybridMixing())

nothing
