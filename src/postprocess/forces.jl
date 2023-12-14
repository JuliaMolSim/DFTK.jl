# This uses the `compute_forces(term, ψ, occ; kwargs...)` function defined by all terms
"""
Compute the forces of an obtained SCF solution. Returns the forces wrt. the fractional
lattice vectors. To get cartesian forces use [`compute_forces_cart`](@ref).
Returns a list of lists of forces (as SVector{3}) in the same order as the `atoms`
and `positions` in the underlying [`Model`](@ref).
"""
@timing function compute_forces(ψ::BlochWaves{T}, occupation; kwargs...) where {T}
    basis = ψ.basis
    # no explicit symmetrization is performed here, it is the
    # responsability of each term to return symmetric forces
    forces_per_term = [compute_forces(term, ψ, occupation; kwargs...)
                       for term in basis.terms]
    sum(filter(!isnothing, forces_per_term))
end

"""
Compute the cartesian forces of an obtained SCF solution in Hartree / Bohr.
Returns a list of lists of forces
`[[force for atom in positions] for (element, positions) in atoms]`
which has the same structure as the `atoms` object passed to the underlying [`Model`](@ref).
"""
function compute_forces_cart(ψ::BlochWaves, occupation; kwargs...)
    forces_reduced = compute_forces(ψ, occupation; kwargs...)
    covector_red_to_cart.(ψ.basis.model, forces_reduced)
end

function compute_forces(scfres)
    compute_forces(scfres.ψ, scfres.occupation; scfres.ρ)
end
function compute_forces_cart(scfres)
    compute_forces_cart(scfres.ψ, scfres.occupation; scfres.ρ)
end
