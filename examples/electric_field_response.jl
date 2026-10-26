# # Electric field responses
#
# This page covers the computation of two related topics
# within density functional perturbation theory (DFPT):
# - The response of the electronic polarization to some perturbation.
# - The response of some quantity to the strength of an applied electric field.
#
# Due to the difficulty of defining the electronic polarization and the
# effect of an electric field for an infinite crystal, these two kinds of
# responses do not fall in the AD-DFPT framework implemented by DFTK.[^SPH25]
# Still, you can compute them with DFTK, with XC functionals up to the meta-GGA
# level of theory.[^PSH26]
#
# [^SPH25]:
#     Schmitz, N. F., Ploumhans, B., & Herbst, M. F. (2026)
#     *Algorithmic differentiation for plane-wave DFT: materials design, error control and learning model parameters.*
#     [npj Computational Materials **12**, 6 (2026)](https://doi.org/10.1038/s41524-025-01880-3).
#     ([Supplementary material and computational scripts](https://github.com/niklasschmitz/ad-dfpt)).
# [^PSH26]:
#     Ploumhans, B., Schmitz, N. F. & Herbst, M. F. (2026).
#     *Density functional perturbation theory of meta-generalized gradient approximations using algorithmic differentiation.*
#     [arXiv:2609.35572 [cond-mat.mtrl-sci]](https://doi.org/10.48550/arXiv.2609.35572).
#
# ## Response of the electronic polarization
# As a first example, we compute Born effective charges (BECs) for
# zincblende boron nitride. The BECs can be defined as
# ```math
# Z^\ast_{s\alpha \beta} = \Omega \frac{\mathrm d P_\beta}{\mathrm d u_{s\alpha}},
# ```
# i.e. the derivative of the polarization in Cartesian direction ``\beta``
# wrt. displacing atom ``s`` in Cartesian direction ``\alpha``.
#
# We consider gallium arsenide, for which the 

using DFTK
using ForwardDiff
using LinearAlgebra
using PseudoPotentialData

lattice = 2.8727u"Å" * [0 1 1.;
                        1 0 1.;
                        1 1 0.]
pseudofamily = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
Ga = ElementPsp(:Ga, pseudofamily)
As = ElementPsp(:As, pseudofamily)
atoms     = [Ga, As]
positions = [zeros(3), ones(3)/4]

model = model_DFT(lattice, atoms, positions; functionals=LDA(), symmetries=false, temperature=1e-3, smearing=Smearing.FermiDirac())
basis = PlaneWaveBasis(model; Ecut=30, kgrid=[6, 6, 6]);
scfres = self_consistent_field(basis, tol=1e-8);

elres = electric_field_responses(scfres);

dielectric_tensor = compute_dielectric_cart(scfres; elres...)

Zstar = compute_Zstar_cart(scfres; elres.electric_field_responses)

nothing
