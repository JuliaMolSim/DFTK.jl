# # Electric field response
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
# As a first example, we compute Born effective charges for
# aluminium arsenide. They can be defined as
# ```math
# Z^\ast_{s\alpha \beta} = \Omega \frac{\mathrm d P_{\mathrm{cart},\beta}}{\mathrm d R_{\mathrm{cart},s\alpha}},
# ```
# i.e. the derivative of the polarization in Cartesian direction ``\beta``
# wrt. displacing atom ``s`` in Cartesian direction ``\alpha``.
#
# For aluminium arsenide, there is only one independent Born effective charge due to symmetries.
# Let's calculate it by computing the polarization response wrt. moving the first
# Ga atom in Cartesian direction x.
#
# Let's already import the required packages and setup the crystal:

using DFTK
using ForwardDiff
using LinearAlgebra
using PseudoPotentialData
using Unitful
using UnitfulAtomic

lattice = 5.4156 * [0 1 1.;
                    1 0 1.;
                    1 1 0.]
pseudofamily = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
Al = ElementPsp(:Al, pseudofamily)
As = ElementPsp(:As, pseudofamily)
atoms     = [Al, As]
positions = [zeros(3), ones(3)/4];

# To speed up the calculation, we will make use of symmetries.
# Let's only use the symmetries that are compatible with the response we will compute,
# by performing symmetry detection with a slightly distorted atom:
symmetries = let
    δpos = [lattice \ [1, 0, 0], [0, 0, 0]]
    model = model_DFT(lattice, atoms, positions + 0.01 * δpos;
                      functionals=PBE())
    model.symmetries
end
println("Number of symmetries: ", length(symmetries))

# We can now build the model and perform the ground state calculation as usual:
model = model_DFT(lattice, atoms, positions;
                  functionals=PBE(), symmetries)
basis = PlaneWaveBasis(model; Ecut=30, kgrid=[2, 2, 2])
scfres = self_consistent_field(basis, tol=1e-6);

# Then, we compute the response wrt. moving the first atom.
# We evaluate the Hamiltonian perturbation by algorithmic differentiation (AD),
# then feed it to the DFPT solver `solve_ΩplusK_split`:

function δHδR_ψ(ε)
    δpos = [lattice \ [1, 0, 0], [0, 0, 0]]
    positions_new = positions + ε * δpos
    ## Note: lattice conversion is required for differentiation
    model_new = Model(model; lattice=convert.(eltype(ε), lattice),
                             positions=positions_new)
    basis_new = PlaneWaveBasis(basis; model=model_new)
    ham_new = Hamiltonian(basis_new; scfres.ρ)
    ham_new * scfres.ψ
end

δHextψ = ForwardDiff.derivative(δHδR_ψ, 0.0)
dfptres = DFTK.solve_ΩplusK_split(scfres, δHextψ; tol=1e-6);

# To evaluate the polarization response, we need another ingredient:
# the derivatives of the wavefunctions $u_{n\mathbf{k}}$ wrt. the wavevector $\mathbf{k}$.
# This is done by calling the DFTK function [`compute_k_derivatives`](@ref):
δψδks = compute_k_derivatives(scfres; tol=1e-8);

# Finally, evaluate the electronic polarization response, and add the ionic contribution.
# Note that [`electronic_polarization_derivative`](@ref) computes the fully reduced polarization
# $\mathbf{P}_\mathrm{el}$, from which the Cartesian polarization can be computed with
# $\mathbf{P}_\mathrm{el,cart} = L / \Omega \mathbf{P}_\mathrm{el}$ where $L$ is the lattice matrix.
dPel_dR = electronic_polarization_derivative(basis, scfres.occupation, dfptres.δψ, δψδks)
ΩdPel_cart_dR = lattice * dPel_dR
Zstar_Al = ΩdPel_cart_dR[1] + charge_ionic(atoms[1])
println("Z*(Al)_xx = ", Zstar_Al, " (units of e)")

# We find a Born effective charge of about 70 because of our **very loose** numerical settings,
# in particular the $\mathbf{k}$-grid which needs to be rather fine for
# electric properties.
#
# Upon running the same calculation with ``E_\mathrm{cut} = 50`` Ha
# and a converged $12 \times 12 \times 12$ $\mathbf{k}$-grid,
# the value $Z^* = 2.14$ is obtained.

# ## Electric field responses
# The Born effective charges can equivalently be computed as the derivative of
# the force wrt. the strength of an applied electric field ``\mathcal E``:
# ```math
# Z^\ast_{s\alpha \beta} = \frac{\mathrm d F_{\mathrm{cart},s\alpha}}{\mathrm d \mathcal E_{\mathrm{cart},\beta}}.
# ```
# For more complex systems this approach is generally preferred as it gives
# the Born effective charges for all the atoms with only three response calculations:
# one per electric field direction.
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
# where ``\sigma`` is the stress in Voigt notation.
# Note that while care is needed to obtain the so-called proper piezoelectric tensor[^Van00]
# from the Berry phase polarization (based on numerical $\mathbf{k}$-derivatives), this is not an
# issue with the linear response formulation (analytical $\mathbf{k}$-derivatives) considered here.
#
# To compute the response to a Cartesian electric field perturbation
# $\delta \mathcal E_\mathrm{cart}$, the following RHS is passed to the DFPT solver:
# ```math
# \mathrm{rhs}_{n\mathbf{k}} = \frac{i}{2\pi} \sum_\alpha \frac{\mathrm d u_{n\mathbf{k}}}{\mathrm d k_\alpha} \delta \mathcal E_\alpha,
# ```
# where the reduced electric field perturbation is $\delta \mathcal E = L^T \delta \mathcal E_\mathrm{cart}$.
#
# Let's compute the response of the aluminium arsenide system above to an electric field
# perturbation along Cartesian direction x. Note that we already computed the
# $\mathbf{k}$-derivatives. First, we assemble the RHS:
#
# [^Van00]:
#     Vanderbilt, D. (2000).
#     *Berry-phase theory of proper piezoelectric response.*
#     [J. Phys. Chem. Solids **61**, 147](https://doi.org/10.1016/S0022-3697(99)00273-5).

δE = lattice' * [1, 0, 0]
rhs = im/2π * (δψδks[1] * δE[1] + δψδks[2] * δE[2] + δψδks[3] * δE[3]);

# Let's also make sure that the symmetries are compatible with the electric field perturbation:
@assert all(symop -> symop.W'δE ≈ δE, symmetries)

# All good, so we can call the DFPT solver:
dfptres_elfield = DFTK.solve_ΩplusK_split(scfres, rhs; tol=1e-6);

# Finally, we compute the change in forces, polarization, and stresses:
dF_dE = ForwardDiff.derivative(0.0) do ε
    compute_forces_cart(basis, scfres.ψ + ε * dfptres_elfield.δψ,
                               scfres.occupation + ε * dfptres_elfield.δoccupation;
                               ρ=scfres.ρ + ε * dfptres_elfield.δρ)
end
dPel_dE = electronic_polarization_derivative(basis, scfres.occupation, dfptres_elfield.δψ, δψδks)
dPel_cart_dE = lattice / model.unit_cell_volume * dPel_dE
dstresses_dE = ForwardDiff.derivative(0.0) do ε
    compute_stresses_cart(basis, scfres.ψ + ε * dfptres_elfield.δψ,
                                 scfres.occupation + ε * dfptres_elfield.δoccupation)
end

println("Z*(Al)_xx = ", dF_dE[1][1] + DFTK.charge_ionic(atoms[1]), " (units of e)")
println("ϵ_xx = ", 1 + 4π * dPel_cart_dE[1], " (units of ϵ₀)")
println("e_14 = ", -DFTK.full_stress_to_voigt(dstresses_dE)[4], " (a.u.)")
println("e_14 = ", -DFTK.full_stress_to_voigt(dstresses_dE)[4] / austrip(1u"C/m^2"), " (C/m^2)")

# We recover the same Born effective charge as with the other route based on
# displacing an atom. Note again that the values here are **severely underconverged**.
#
# When increasing $E_\mathrm{cut}$ to $50$ Ha and the $\mathbf{k}$-grid
# to $12 \times 12 \times 12$, the following numbers are obtained:
# - Born effective charge: $Z^* = 2.14$
# - Dielectric constant: $\epsilon_{xx} = 9.5 \epsilon_0$
# - Clamped-ion piezoelectric constant: $e_{14} = -0.69$ C/m²

