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
# ## Response of the polarization to an ionic perturbation
# As a first example, we compute Born effective charges (BECs) for
# gallium arsenide. The BECs can be defined as
# TODO problem: two conflicting meanings for "u"
# ```math
# Z^\ast_{s\alpha \beta} = \Omega \frac{\mathrm d P_{\mathrm{cart},\beta}}{\mathrm d u_{\mathrm{cart},s\alpha}},
# ```
# i.e. the derivative of the polarization in Cartesian direction ``\beta``
# wrt. displacing atom ``s`` in Cartesian direction ``\alpha``.
#
# For gallium arsenide, there is only one independent BEC due to symmetries.
# Let's calculate it by computing the polarization response wrt. moving the first
# Ga atom in Cartesian direction x.
# We first perform a ground state calculation:

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
atoms     = [Ga, As]
positions = [zeros(3), ones(3)/4]
model = model_DFT(lattice, atoms, positions;
                  ## TODO: k derivative currently requires symmetries=false, fix this!
                  functionals=PBE(), symmetries=false)
basis = PlaneWaveBasis(model; Ecut=30, kgrid=[2, 2, 2])
scfres = self_consistent_field(basis, tol=1e-6);

# Then, we compute the response wrt. moving the first atom by evaluating
# the Hamiltonian perturbation by AD and then feeding it to the
# DFPT solver `solve_ΩplusK_split`:

function δHδu_ψ(ε)
    δpos = [lattice \ [1, 0, 0], [0, 0, 0]]
    positions_new = positions + ε * δpos
    ## Note: lattice conversion is required for differentiation
    model_new = Model(model; lattice=convert.(eltype(ε), lattice),
                             positions=positions_new)
    basis_new = PlaneWaveBasis(basis; model=model_new)
    ham_new = Hamiltonian(basis_new; scfres.ρ)
    ham_new * scfres.ψ
end

δHextψ = ForwardDiff.derivative(δHδu_ψ, 0.0)
dfptres = DFTK.solve_ΩplusK_split(scfres, δHextψ; tol=1e-6);

# To evaluate the polarization response, we need another ingredient:
# the derivatives of the wavefunctions $u_{n\mathbf{k}}$ wrt. the wavevector $\mathbf{k}$.
# This is done by calling the DFTK function `compute_δψδks`:
δψδks = DFTK.compute_δψδks(scfres; tol=1e-6);

# Finally, evaluate the electronic polarization response, and add the ionic contribution.
# Note `electronic_polarization_derivative` computes the fully reduced polarization
# $\mathbf{P}_\mathrm{el}$, from which the Cartesian polarization can be computed with
# $\mathbf{P}_\mathrm{el,cart} = L / \Omega \mathbf{P}_\mathrm{el}$ where $L$ is the lattice matrix.
dPel_du = DFTK.electronic_polarization_derivative(basis, scfres.occupation, dfptres.δψ, δψδks)
ΩdPel_cart_du = lattice * dPel_du
Zstar_Ga = ΩdPel_cart_du[1] + charge_ionic(atoms[1])
println("Z*(Ga)_xx = ", Zstar_Ga, " (units of e)")

# TODO: add remark about value being severely underconverged wrt. k,
# and mention the obtained value for converged Ecut and kgrid
# TODO: add comment explaining that [2,2,2] gives completely terrible results
# and also provide reasonably converged values (will need to run with MPI)

# ## Electric field responses
# The BECs can equivalently be computed
# as the derivative of the force wrt. the strength of an applied electric field ``\bm{\mathcal E}``:
# ```math
# Z^\ast_{s\alpha \beta} = \frac{\mathrm d F_{\mathrm{cart},s\alpha}}{\mathrm d \mathcal E_{\mathrm{cart},\beta}}.
# ```
# This only requires one response calculation per field direction,
# and gives the BECs of all atoms at once.
#
# Other notable quantities that can be obtained as derivatives wrt. an applied electric field
# are the (infinite frequency) dielectric constant:
# ```math
# \epsilon^\infty_{\alpha\beta} = \delta_{\alpha\beta} + 4\pi \frac{\mathrm d P_{\mathrm{cart},\alpha}}{\mathrm d \mathcal E_{\mathrm{cart},\beta}}
# ```
# and the (clamped-ion) piezoelectric tensor
# ```math
# e_{\alpha j} = -\frac{\mathrm d \sigma_j}{\mathrm d \mathcal E_{\mathrm{cart},\alpha}}
# ```
# where ``\eta`` is the strain.
# Note that while care is needed to obtain the so-called proper piezoelectric tensor[^Van00]
# from the Berry phase polarization (based on numerical k-derivatives), this is not an
# issue with the linear response formulation (analytical k-derivatives) considered here.
# with the linear response formalism here this is not an issue.
#
# To compute the response to a Cartesian electric field perturbation $\delta \mathcal E_\mathrm{cart}$,
# the following RHS is passed to the DFPT solver:
# ```math
# \mathrm{rhs}_{n\mathbf{k}} = \frac{i}{2\pi} \sum_\alpha \frac{\mathrm d u_{n\mathbf{k}}}{\mathrm d k_\alpha} \delta \mathcal E_\alpha,
# ```
# where the reduced electric field is $E = L^T E_\mathrm{cart}$.
#
# Let's compute the response of the gallium arsenide system above to an electric field
# perturbation along Cartesian direction x. Note that we already computed the
# $\mathbf{k}$-derivatives. First, we assemble the RHS:
# TODO: mathbf usage seems rather inconsistent?
#
# [^Van00]:
#     Vanderbilt, D. (2000).
#     *Berry-phase theory of proper piezoelectric response.*
#     [J. Phys. Chem. Solids **61**, 147](https://doi.org/10.1016/S0022-3697(99)00273-5).

δE = lattice' * [1, 0, 0]
rhs = im/2π * (δψδks[1] * δE[1] + δψδks[2] * δE[2] + δψδks[3] * δE[3])

# Then call the DFPT solver:
dfptres_elfield = DFTK.solve_ΩplusK_split(scfres, rhs; tol=1e-6)

# Finally, we compute the change in forces, polarization, and stresses:
# TODO: inconsistency between d and delta
dF = ForwardDiff.derivative(0.0) do ε
    compute_forces_cart(basis, scfres.ψ + ε * dfptres_elfield.δψ,
                               scfres.occupation + ε * dfptres_elfield.δoccupation;
                               ρ=scfres.ρ + ε * dfptres_elfield.δρ)
end
dPel = DFTK.electronic_polarization_derivative(basis, scfres.occupation, dfptres_elfield.δψ, δψδks)
dPel_cart = lattice / model.unit_cell_volume * dPel
dstresses = ForwardDiff.derivative(0.0) do ε
    compute_stresses_cart(basis, scfres.ψ + ε * dfptres_elfield.δψ,
                                 scfres.occupation + ε * dfptres_elfield.δoccupation;
                                 # TODO: default these to nothing?
                                 eigenvalues=nothing, εF=nothing)
end

println("Z*(Ga)_xx = ", dF[1][1] + DFTK.charge_ionic(atoms[1]), " (units of e)")
println("ϵ_xx = ", 1 + 4π * dPel_cart[1], " (units of ϵ₀)")
println("e_14 = ", -DFTK.full_stress_to_voigt(dstresses)[4], " (a.u.)")
println("e_14 = ", -DFTK.full_stress_to_voigt(dstresses)[4] / austrip(1u"C/m^2"), " (C/m^2)")

# TODO: again, severely underconverged

