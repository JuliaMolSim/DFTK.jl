# # Polarizability using automatic differentiation
#
# Simple example for computing properties using (forward-mode)
# automatic differentation.
# For a more classical approach and more details about computing polarizabilities,
# see [Polarizability by linear response](@ref).

using DFTK
using LinearAlgebra
using ForwardDiff

## Construct PlaneWaveBasis given a particular electric field strength
## Again we take the example of a Helium atom.
function make_basis(ε::T; a=10., Ecut=30) where T
    lattice=T(a) * I(3)  # lattice is a cube of ``a`` Bohrs
    He = ElementPsp(:He, psp=load_psp("hgh/lda/He-q2"))
    atoms = [He => [[1/2; 1/2; 1/2]]]  # Helium at the center of the box

    model = model_DFT(lattice, atoms, [:lda_x, :lda_c_vwn];
                      extra_terms=[ExternalFromReal(r -> -ε * (r[1] - a/2))],
                      symmetries=false)
    PlaneWaveBasis(model; Ecut, kgrid=[1, 1, 1])  # No k-point sampling on isolated system
end

## dipole moment of a given density (assuming the current geometry)
function dipole(basis, ρ)
    @assert isdiag(basis.model.lattice)
    a  = basis.model.lattice[1, 1]
    rr = [a * (r[1] - 1/2) for r in r_vectors(basis)]
    sum(rr .* ρ) * basis.dvol
end

## Function to compute the dipole for a given field strength
function compute_dipole(ε; tol=1e-8, kwargs...)
    scfres = self_consistent_field(make_basis(ε; kwargs...), tol=tol)
    dipole(scfres.basis, scfres.ρ)
end;

# With this in place we can compute the polarizability from finite differences
# (just like in the previous example):
polarizability_fd = let
    ε = 0.01
    (compute_dipole(ε) - compute_dipole(0.0)) / ε
end

# ## Forward-mode implicit differentiation
#
# Right now DFTK has no out-of-the-box support for implicit differentiation through the SCF.
# However one can easily work around this as follows. We keep both a non-dual basis
# and a basis including duals for easy bookkeeping (but redundant computation ...).

function self_consistent_field_dual(basis::PlaneWaveBasis, basis_dual::PlaneWaveBasis{T};
                                    kwargs...) where T <: ForwardDiff.Dual
    scfres = self_consistent_field(basis; kwargs...)
    ψ = DFTK.select_occupied_orbitals(basis, scfres.ψ)
    filled_occ = DFTK.filled_occupation(basis.model)
    n_spin = basis.model.n_spin_components
    n_bands = div(basis.model.n_electrons, n_spin * filled_occ)
    occupation = [filled_occ * ones(n_bands) for _ in basis.kpoints]

    ## promote everything eagerly to Dual numbers
    occupation_dual = [T.(occupation[1])]
    ψ_dual = [Complex.(T.(real(ψ[1])), T.(imag(ψ[1])))]
    ρ_dual = compute_density(basis_dual, ψ_dual, occupation_dual)

    _, δH = energy_hamiltonian(basis_dual, ψ_dual, occupation_dual; ρ=ρ_dual)
    δHψ = δH * ψ_dual
    δHψ = [ForwardDiff.partials.(δHψ[1], 1)]
    δψ = DFTK.solve_ΩplusK(basis, ψ, -δHψ, occupation)
    δρ = DFTK.compute_δρ(basis, ψ, δψ, occupation)
    ρ = ForwardDiff.value.(ρ_dual)
    ψ, ρ, δψ, δρ
end;

# This function is now used in the following to provide a dual version
# for the compute_dipole function:
function compute_dipole(ε::ForwardDiff.Dual; tol=1e-8, kwargs...)
    T = ForwardDiff.tagtype(ε)
    basis = make_basis(ForwardDiff.value(ε); kwargs...)
    basis_dual = make_basis(ε; kwargs...)
    ψ, ρ, δψ, δρ = self_consistent_field_dual(basis, basis_dual; tol)
    ρ_dual = ForwardDiff.Dual{T}.(ρ, δρ)
    dipole(basis_dual, ρ_dual)
end;

# This setup allows to compute the polarizability via automatic differentiation:
polarizability = ForwardDiff.derivative(compute_dipole, 0.0)
println()
println("Polarizability via ForwardDiff:       $polarizability")
println("Polarizability via finite difference: $polarizability_fd")
