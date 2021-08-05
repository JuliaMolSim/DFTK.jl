using PyCall
using DFTK
using Unitful
using UnitfulAtomic

kgrid = [1, 1, 1]
Ecut  = 10

lattice = load_lattice("./NaCl.in")
atoms   = load_atoms("./NaCl.in")
atoms = [ElementPsp(el.symbol, psp=load_psp(el.symbol, functional="lda")) => position
         for (el, position) in atoms]

model = model_LDA(lattice, atoms)
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)

scfres = self_consistent_field(basis, tol=1e-8);

@show compute_dipole_moment(scfres)

dipole_ABINIT = [  0.309208E+01 ,   -0.192893E-01,    -0.192893E-01]
dipole_GPAW = [-1.680941, -0.000000, -0.000000] .* austrip(1u"Å")  #DFTK.units.Å

@show dipole_ABINIT
@show dipole_GPAW
nothing
