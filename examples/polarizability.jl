# # Polarizability by linear response
#
# We compute the polarizability of a Helium atom. The polarizability
# is defined as the change in dipole moment
# ```math
# μ = ∫ r ρ(r) dr
# ```
# with respect to a small uniform electric field ``E = -x``.
#
# We compute this in two ways: first by finite differences (applying a
# finite electric field), then by linear response. Note that DFTK is
# not really adapted to isolated atoms because it uses periodic
# boundary conditions. Nevertheless we can simply embed the Helium
# atom in a large enough box (although this is computationally wasteful).
#
# As in other tests, this is not fully converged, convergence
# parameters were simply selected for fast execution on CI,

using DFTK
using LinearAlgebra

a = 10.
lattice = a * I(3)  # cube of ``a`` bohrs
## Helium at the center of the box
atoms     = [ElementPsp(:He; psp=load_psp("hgh/lda/He-q2"))]
positions = [[1/2, 1/2, 1/2]]


kgrid = [1, 1, 1]  # no k-point sampling for an isolated system
Ecut = 30
tol = 1e-8

## dipole moment of a given density (assuming the current geometry)
function dipole(basis, ρ)
    rr = [(r[1] - a/2) for r in r_vectors_cart(basis)]
    sum(rr .* ρ) * basis.dvol
end;

# ## Using finite differences
# We first compute the polarizability by finite differences.
# First compute the dipole moment at rest:
model = model_LDA(lattice, atoms, positions; symmetries=false)
basis = PlaneWaveBasis(model; Ecut, kgrid)
res   = self_consistent_field(basis; tol)
μref  = dipole(basis, res.ρ)

# Then in a small uniform field:
ε = .01
model_ε = model_LDA(lattice, atoms, positions;
                    extra_terms=[ExternalFromReal(r -> -ε * (r[1] - a/2))],
                    symmetries=false)
basis_ε = PlaneWaveBasis(model_ε; Ecut, kgrid)
res_ε   = self_consistent_field(basis_ε; tol)
με = dipole(basis_ε, res_ε.ρ)

#-
polarizability = (με - μref) / ε

println("Reference dipole:  $μref")
println("Displaced dipole:  $με")
println("Polarizability :   $polarizability")

# The result on more converged grids is very close to published results.
# For example [DOI 10.1039/C8CP03569E](https://doi.org/10.1039/C8CP03569E)
# quotes **1.65** with LSDA and **1.38** with CCSD(T).

# ## Using linear response
# Now we use linear response to compute this analytically; we refer to standard
# textbooks for the formalism. In the following, ``χ_0`` is the
# independent-particle polarizability, and ``K`` the
# Hartree-exchange-correlation kernel. We denote with ``δV_{\rm ext}`` an external
# perturbing potential (like in this case the uniform electric field). Then:
# ```math
# δρ = χ_0 δV = χ_0 (δV_{\rm ext} + K δρ),
# ```
# which implies
# ```math
# δρ = (1-χ_0 K)^{-1} χ_0 δV_{\rm ext}.
# ```
# From this we identify the polarizability operator to be ``χ = (1-χ_0 K)^{-1} χ_0``.
# Numerically, we apply ``χ`` to ``δV = -x`` by solving a linear equation
# (the Dyson equation) iteratively.

using KrylovKit

## Apply ``(1- χ_0 K)``
function dielectric_operator(δρ)
    δV = apply_kernel(basis, δρ; res.ρ)
    χ0δV = apply_χ0(res, δV)
    δρ - χ0δV
end

## `δVext` is the potential from a uniform field interacting with the dielectric dipole
## of the density.
δVext = [-(r[1] - a/2) for r in r_vectors_cart(basis)]
δVext = cat(δVext; dims=4)

## Apply ``χ_0`` once to get non-interacting dipole
δρ_nointeract = apply_χ0(res, δVext)

## Solve Dyson equation to get interacting dipole
δρ = linsolve(dielectric_operator, δρ_nointeract, verbosity=3)[1]

println("Non-interacting polarizability: $(dipole(basis, δρ_nointeract))")
println("Interacting polarizability:     $(dipole(basis, δρ))")

# As expected, the interacting polarizability matches the finite difference
# result. The non-interacting polarizability is higher.
