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

using ComponentArrays
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

# dpos = [lattice \ [1, 0, 0],
#         [0, 0, 0]]

# dP = ForwardDiff.derivative(ε -> position_to_ΩP(positions0 + ε * dpos), 0.0)
# @show dP

# TODO: add comment explaining that [2,2,2] gives completely terrible results
# and also provide reasonably converged values (will need to run with MPI)

# ## Response to an electric field
# Thanks to the electric enthalpy formulation, the BECs can equivalently be computed
# as the derivative of the force wrt. the strength of an applied electric field ``\bm{\mathcal E}``:
# ```math
# Z^\ast_{s\alpha \beta} = \frac{\mathrm d F_{s\alpha}}{\mathrm d \mathcal E_\beta}.
# ```
# This only requires one response calculation per field direction,
# and gives the BECs of all atoms at once.
#
# Other notable quantities that can be obtained as derivatives wrt. an applied electric field
# are the (infinite frequency) dielectric constant:
# ```math
# \epsilon^\infty_{\alpha\beta} = \delta_{\alpha\beta} + 4\pi \frac{\mathrm d P_\alpha}{\mathrm d \mathcal E_\beta}
# ```
# and the (clamped-ion) piezoelectric tensor
# TODO: cleanup LLM slop below
# ```math
# e_{\alpha jk} = \frac{\mathrm d P_\alpha}{\mathrm d \eta_{jk}}
# ```
# where ``\eta`` is the strain.
# Some care is needed here: naively differentiating the Cartesian polarization
# ``\mathbf P = L \mathbf P_\mathrm{red} / \Omega`` wrt. strain also picks up the change
# of ``L / \Omega``, which yields the so-called *improper* piezoelectric tensor.
# The physically meaningful *proper* tensor[^Van00] only differentiates the reduced
# polarization:
# ```math
# e_{\alpha jk} = \frac{1}{\Omega} \sum_\beta L_{\alpha\beta}
#                  \frac{\mathrm d P_{\mathrm{red},\beta}}{\mathrm d \eta_{jk}}.
# ```
# Since DFTK writes the electric enthalpy as
# ``-\bm{\mathcal E}_\mathrm{red} \cdot \mathbf P_\mathrm{red}`` with the reduced field
# ``\bm{\mathcal E}_\mathrm{red} = L^T \bm{\mathcal E}`` kept fixed under strain,
# the field derivative of the stress ``\sigma_{jk} = \frac{1}{\Omega} \frac{\partial F}{\partial \eta_{jk}}``
# yields exactly the proper tensor:
# ```math
# \frac{\mathrm d \sigma_{jk}}{\mathrm d \mathcal E_\alpha}
# = \sum_\beta L_{\alpha\beta} \frac{\mathrm d \sigma_{jk}}{\mathrm d \mathcal E_{\mathrm{red},\beta}}
# = -\frac{1}{\Omega} \sum_\beta L_{\alpha\beta} \frac{\mathrm d P_{\mathrm{red},\beta}}{\mathrm d \eta_{jk}}
# = -e_{\alpha jk}.
# ```
#
# [^Van00]:
#     Vanderbilt, D. (2000).
#     *Berry-phase theory of proper piezoelectric response.*
#     [J. Phys. Chem. Solids **61**, 147](https://doi.org/10.1016/S0022-3697(99)00273-5).

function field_response(E)
    T = eltype(E)
    field = DFTK.ElectricField(E, DFTK.LinearResponsePolarization())
    model = model_DFT(convert.(T, lattice), atoms, positions0;
                      functionals=PBE(), symmetries=false, extra_terms=[field])
    basis = PlaneWaveBasis(model; Ecut=30, kgrid=[2, 2, 2]);
    scfres = self_consistent_field(basis, tol=1e-6);

    # Put all quantities in a component array such that ForwardDiff
    # will be able to unwrapp all the components at once:
    ComponentArray(;
        # forces[α, s] = F_{sα}
        forces = stack(compute_forces_cart(scfres)),
        P      = DFTK.total_polarization_cart(scfres, DFTK.LinearResponsePolarization()),
        stress = compute_stresses_cart(scfres),
    )
end

## DFTK natively operates on the reduced electric field:
dE = lattice' * [1, 0, 0]
dresponse = ForwardDiff.derivative(ε -> field_response(zeros(3) + ε * dE), 0.0)

@show dresponse.forces[1, 1]
@show dresponse.P
ϵ_11 = 1 + 4π * dresponse.P[1]
@show ϵ_11
## Proper clamped-ion piezoelectric tensor e_{1jk} = -dσ_jk/dE_1
e_1 = -dresponse.stress
@show e_1
# TODO: would be nice to have in C/m^2, and only the 14 component. Maybe keep voigt notation

# elres = electric_field_responses(scfres);

# dielectric_tensor = compute_dielectric_cart(scfres; elres...)

# Zstar = compute_Zstar_cart(scfres; elres.electric_field_responses)

nothing
