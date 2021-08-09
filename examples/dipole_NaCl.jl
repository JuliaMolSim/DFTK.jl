using PyCall
using DFTK
using Unitful
using UnitfulAtomic

kgrid = [1, 1, 1]
Ecut  = 10

# lattice = load_lattice("./NaCl.in")
# atoms   = load_atoms("./NaCl.in")
lattice = diagm([20., 10, 10])
Na = ElementPsp(:Na, psp=load_psp("hgh/lda/na-q1.hgh"))
Cl = ElementPsp(:Cl, psp=load_psp("hgh/lda/cl-q7.hgh"))
atoms   = [Na => [[-1.1810788331E-01,0,0] ], Cl => [[1.1810788331E-01,0,0] ]]

model = model_LDA(lattice, atoms)
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid, fft_size=[84, 42, 42])

scfres = self_consistent_field(basis, tol=1e-8);


ndipmom = DFTK.nuclear_dipole_moment(model)
ccharge = DFTK.center_of_charge(model.atoms)
dipmom  = compute_dipole_moment(scfres, center=ccharge)
@show ndipmom * model.unit_cell_volume
@show dipmom * model.unit_cell_volume
@show ccharge

ccharge_ABINIT = [8.85809E-02,  0.00000E+00,  0.00000E+00]
ndipmom_ABINIT = [0.141729E+02     0.000000E+00     0.000000E+00]
dipole_ABINIT = [  0.309208E+01 ,   -0.192893E-01,    -0.192893E-01]
dipole_GPAW = austrip.((4.196235, 0.000000, -0.000000) .* 1u"Å")

@show ccharge_ABINIT
@show ndipmom_ABINIT
@show dipole_ABINIT
@show dipole_GPAW
nothing
