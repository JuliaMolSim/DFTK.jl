# This uses the `compute_forces(term, ψ, occ; kwargs...)` function defined by all terms
"""
Compute the forces of an obtained SCF solution. Returns the forces wrt. the fractional
lattice vectors. To get Cartesian forces use [`compute_forces_cart`](@ref).
Returns a list of lists of forces (as SVector{3}) in the same order as the `atoms`
and `positions` in the underlying [`Model`](@ref).
"""
@timing function compute_forces(basis::PlaneWaveBasis{T}, ψ, occupation; kwargs...) where {T}
    # no explicit symmetrization is performed here, it is the
    # responsability of each term to return symmetric forces
    forces_per_term = [compute_forces(term, basis, ψ, occupation; kwargs...)
                       for term in basis.terms]
    sum(filter(!isnothing, forces_per_term))
end

"""
Compute the Cartesian forces of an obtained SCF solution in Hartree / Bohr.
Returns a list of lists of forces
`[[force for atom in positions] for (element, positions) in atoms]`
which has the same structure as the `atoms` object passed to the underlying [`Model`](@ref).
"""
function compute_forces_cart(basis::PlaneWaveBasis, ψ, occupation; kwargs...)
    forces_reduced = compute_forces(basis, ψ, occupation; kwargs...)
    covector_red_to_cart.(basis.model, forces_reduced)
end

function compute_forces(scfres)
    compute_forces(scfres.basis, scfres.ψ, scfres.occupation; scfres.ρ)
end
function compute_forces_cart(scfres)
    compute_forces_cart(scfres.basis, scfres.ψ, scfres.occupation; scfres.ρ)
end
