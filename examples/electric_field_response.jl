using DFTK
using ForwardDiff
using LinearAlgebra
using PseudoPotentialData

a = 6.83  # zincblende BN lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                    [1 0 1.];
                    [1 1 0.]]
pseudofamily = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
B  = ElementPsp(:B, pseudofamily)
N  = ElementPsp(:N, pseudofamily)
atoms     = [B, N]
positions = [ones(3)/8, -ones(3)/8]

model = model_DFT(lattice, atoms, positions; functionals=LDA(), symmetries=false, temperature=1e-3, smearing=Smearing.FermiDirac())
basis = PlaneWaveBasis(model; Ecut=30, kgrid=[6, 6, 6]);
scfres = self_consistent_field(basis, tol=1e-8);

elres = electric_field_responses(scfres);

dielectric_tensor = compute_dielectric_cart(scfres; elres...)

Zstar = compute_Zstar_cart(scfres; elres.electric_field_responses)

nothing
