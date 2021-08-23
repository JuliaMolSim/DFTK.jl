# # Polarizability by linear response
#
# We compute the polarizability of a Helium atom. The polarizability
# is defined as the change in dipole moment
# ```math
# \mu = \int r ρ(r) dr
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
He = ElementPsp(:He, psp=load_psp("hgh/lda/He-q2"))
atoms = [He => [[1/2; 1/2; 1/2]]]  # Helium at the center of the box

kgrid = [1, 1, 1]  # no kpoint sampling for an isolated system
Ecut = 30
tol = 1e-8

## dipole moment of a given density (assuming the current geometry)
function dipole(basis, ρ)
    rr = [a * (r[1] - 1/2) for r in r_vectors(basis)]
    d = sum(rr .* ρ) * basis.dvol
end;

# ## Polarizability by finite differences
# We first compute the polarizability by finite differences.
# First compute the dipole moment at rest:
model = model_LDA(lattice, atoms; symmetries=false)
basis = PlaneWaveBasis(model; Ecut, kgrid)
res = self_consistent_field(basis, tol=tol)
μref = dipole(basis, res.ρ)

# Then in a small uniform field:
ε = .01
model_ε = model_LDA(lattice, atoms; extra_terms=[ExternalFromReal(r -> -ε * (r[1] - a/2))],
                    symmetries=false)
basis_ε = PlaneWaveBasis(model_ε; Ecut, kgrid)
res_ε = self_consistent_field(basis_ε, tol=tol)
με = dipole(basis_ε, res_ε.ρ)

#-
polarizability = (με - μref) / ε

println("Reference dipole:  $μref")
println("Displaced dipole:  $με")
println("Polarizability :   $polarizability")

# The result on more converged grids is very close to published results.
# For example [DOI 10.1039/C8CP03569E](https://doi.org/10.1039/C8CP03569E)
# quotes **1.65** with LSDA and **1.38** with CCSD(T).

# ## Polarizability by linear response
# Now we use linear response to compute this analytically; we refer to standard
# textbooks for the formalism. In the following, ``\chi_0`` is the
# independent-particle polarizability, and ``K`` the
# Hartree-exchange-correlation kernel. We denote with ``\delta V_{\rm ext}`` an external
# perturbing potential (like in this case the uniform electric field). Then:
# ```math
# \delta\rho = \chi_0 \delta V = \chi_0 (\delta V_{\rm ext} + K \delta\rho),
# ```
# which implies
# ```math
# \delta\rho = (1-\chi_0 K)^-1 \chi_0 \delta V_{\rm ext}.
# ```
# From this we identify the polarizability operator to be ``\chi = (1-\chi_0 K)^{-1} \chi_0``.
# Numerically, we apply ``\chi`` to ``\delta V = -x`` by solving a linear equation
# (the Dyson equation) iteratively.

using KrylovKit

## Apply (1- χ0 K)
function dielectric_operator(δρ)
    δV = apply_kernel(basis, δρ; ρ=res.ρ)
    χ0δV = apply_χ0(res.ham, res.ψ, res.εF, res.eigenvalues, δV)
    δρ - χ0δV
end

## δVext is the potential from a uniform field interacting with the dielectric dipole
## of the density.
δVext = [-a * (r[1] - 1/2) for r in r_vectors(basis)]
δVext = cat(δVext; dims=4)

## Apply χ0 once to get non-interacting dipole
δρ_nointeract = apply_χ0(res.ham, res.ψ, res.εF, res.eigenvalues, δVext)

## Solve Dyson equation to get interacting dipole
δρ = linsolve(dielectric_operator, δρ_nointeract, verbosity=3)[1]

println("Non-interacting polarizability: $(dipole(basis, δρ_nointeract))")
println("Interacting polarizability:     $(dipole(basis, δρ))")

# As expected, the interacting polarizability matches the finite difference
# result. The non-interacting polarizability is higher.

#=============================================================================#
# # Dipole moment using ForwardDiff

using ForwardDiff

function obj(ε)
    basis_ε = make_basis(ε)
    res_ε = self_consistent_field(basis_ε, tol=tol)
    dipole(basis_ε, res_ε.ρ)
end
# goal: obj'(0)

function make_basis(ε, lattice)
    # model_ε = model_LDA(lattice, atoms; extra_terms=[ExternalFromReal(r -> -ε * (r[1] - a/2))],
    #                 symmetries=false) # Fallback functional for lda_xc_teter93 not implemented.
    model_ε = model_DFT(lattice, atoms, [:lda_x, :lda_c_vwn];  
                        extra_terms=[ExternalFromReal(r -> -ε * (r[1] - a/2))],
                        symmetries=false)
    basis_ε = PlaneWaveBasis(model_ε; Ecut, kgrid)
end
make_basis(ε) = make_basis(ε, lattice)

# workaround: enforce the Model to promote its type from Float64 to Dual, by promoting the lattice too
make_basis(ε::T) where T <: ForwardDiff.Dual = make_basis(ε, T.(lattice))

# forward mode implicit differentiation of SCF

# Approach 0: keep both a non-dual basis, and a basis including Duals for easy bookkeeping (but redundant computation)
function self_consistent_field_dual(basis::PlaneWaveBasis, basis_dual::PlaneWaveBasis{T}; kwargs...) where T <: ForwardDiff.Dual
    scfres = self_consistent_field(basis; kwargs...)
    # ψ = scfres.ψ
    # occupation = scfres.occupation
    ψ = DFTK.select_occupied_orbitals(basis, scfres.ψ)
    filled_occ = DFTK.filled_occupation(basis.model)
    n_spin = basis.model.n_spin_components
    n_bands = div(basis.model.n_electrons, n_spin * filled_occ)
    # number of kpoints and occupation
    Nk = length(basis.kpoints)
    occupation = [filled_occ * ones(n_bands) for ik = 1:Nk]

    # promote everything eagerly to Dual numbers
    # TODO figure out how to get by without this
    occupation_dual = [T.(occupation[1])]
    ψ_dual = [Complex.(T.(real(ψ[1])), T.(imag(ψ[1])))]
    ρ_dual = compute_density(basis_dual, ψ_dual, occupation_dual)

    _, δH = energy_hamiltonian(basis_dual, ψ_dual, occupation_dual; ρ=ρ_dual)
    δHψ = δH * ψ_dual
    δHψ = [ForwardDiff.partials.(δHψ[1], 1)] # keep only partial components of duals
    δψ = DFTK.solve_ΩplusK(basis, ψ, -δHψ, occupation)
    δρ = DFTK.compute_δρ(basis, ψ, δψ, occupation)
    ρ = ForwardDiff.value.(ρ_dual)
    ψ, ρ, δψ, δρ
end

# TODO next steps.
# - [ ] somehow verify δψ
# - [x] compute δρ from δψ
# - [x] diff through dipole
# - [x] compose the thing into obj'(ε)
# - [x] compare against finite difference

function obj(ε::ForwardDiff.Dual)
    T = ForwardDiff.tagtype(ε)
    basis = make_basis(ForwardDiff.value(ε))
    basis_dual = make_basis(ε)
    ψ, ρ, δψ, δρ = self_consistent_field_dual(basis, basis_dual; tol=tol)
    ρ_dual = ForwardDiff.Dual{T}.(ρ, δρ)
    dipole(basis_dual, ρ_dual)
end

# obj(ForwardDiff.Dual{:qwerty}(0.0, 1.0))
ForwardDiff.derivative(obj, 0.0) # 1.7725352397017748

let d = 1e-4
    (obj(d) - obj(0.0)) / d
end # 1.772091814276788
