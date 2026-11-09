using DFTK
using ForwardDiff
using LinearAlgebra
using PseudoPotentialData
using Unitful
using UnitfulAtomic

a = 6.1905
c = 9.9886
lattice = [a    -a/2    0.0;
           0.0  a*√3/2  0.0;
           0.0  0.0     c]
pseudofamily = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
Zn = ElementPsp(:Zn, pseudofamily)
O = ElementPsp(:O, pseudofamily)
atoms = [Zn, Zn, O, O]
positions = [[1/3, 2/3, 0],
             [2/3, 1/3, 1/2],
             [1/3, 2/3, 0.3792],
             [2/3, 1/3, 1/2 + 0.3792]]

model = model_DFT(lattice, atoms, positions; functionals=PBE())
basis = PlaneWaveBasis(model; Ecut=20, kgrid=[2, 2, 2])
scfres = self_consistent_field(basis; tol=1e-6)

scfres_unfolded = DFTK.unfold_bz(scfres; skip_hamiltonian=true)

P = DFTK.total_polarization_cart(scfres_unfolded, DFTK.BerryPhasePolarization())

P_SI = auconvert.(u"C/m^2", P)
