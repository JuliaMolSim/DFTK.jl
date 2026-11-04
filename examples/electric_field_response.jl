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
#     Schmitz, N. F., Ploumhans, B. & Herbst, M. F. (2026)
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
# gallium arsenide. The BECs can be defined as
# ```math
# Z^\ast_{s\alpha \beta} = \Omega \frac{\mathrm d P_\beta}{\mathrm d u_{s\alpha}},
# ```
# i.e. the derivative of the polarization in Cartesian direction ``\beta``
# wrt. displacing atom ``s`` in Cartesian direction ``\alpha``.

using DFTK
using ForwardDiff
using LinearAlgebra
using PseudoPotentialData
using Unitful
using UnitfulAtomic

lattice = 5.4286 * [0 1 1.;
                    1 0 1.;
                    1 1 0.]
pseudofamily = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
Ga = ElementPsp(:Ga, pseudofamily)
As = ElementPsp(:As, pseudofamily)
atoms      = [Ga, As]
positions0 = [zeros(3), ones(3)/4]

# Temporary: use ZnO
# lattice = [6.190456849814106 -3.0952284154584224 -0.0;
#            0.0 5.361092884478024 0.0;
#            -0.0 0.0 9.988640102207459]
# Zn = ElementPsp(:Zn, pseudofamily)
# O = ElementPsp(:O, pseudofamily)
# atoms = [Zn, Zn, O, O]
# positions0 = [[0.3333333323157854, 0.6666666666666667, -0.017092629605625867],
#               [0.6666666646315706, 0.33333333333333337, 0.4829073694484365],
#               [0.3333333323157854, 0.6666666666666667, 0.36209286326168183],
#               [0.6666666646315706, 0.33333333333333337, 0.862092862315744]]


function position_to_ΩP(positions)
    T = eltype(eltype(positions))
    model = model_DFT(convert.(T, lattice), atoms, positions;
                      # TODO: k derivative currently requires symmetries=false, fix this!
                      functionals=PBE(), symmetries=false)
    basis = PlaneWaveBasis(model; Ecut=30, kgrid=[2, 2, 2]);
    # basis = PlaneWaveBasis(model; Ecut=30, kgrid=[6, 6, 6]);
    scfres = self_consistent_field(basis, tol=1e-6);

    # total_polarization_cart(scfres, LinearResponsePolarization())
    model.lattice * DFTK.total_polarization(scfres, DFTK.LinearResponsePolarization())
end

dpos = [lattice \ [1, 0, 0],
        [0, 0, 0]]

dP = ForwardDiff.derivative(ε -> position_to_ΩP(positions0 + ε * dpos), 0.0)
@show dP

# TODO: add comment explaining that [2,2,2] gives completely terrible results
# and also provide reasonably converged values (will need to run with MPI)

# elres = electric_field_responses(scfres);

# dielectric_tensor = compute_dielectric_cart(scfres; elres...)

# Zstar = compute_Zstar_cart(scfres; elres.electric_field_responses)

nothing
