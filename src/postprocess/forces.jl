# This uses the `compute_forces(term, ψ, occ; kwargs...)` function defined by all terms
"""
Compute the forces of an obtained SCF solution. Returns the forces wrt. the fractional
lattice vectors. To get Cartesian forces use [`compute_forces_cart`](@ref).
Returns a list of lists of forces (as SVector{3}) in the same order as the `atoms`
and `positions` in the underlying [`Model`](@ref).

!!! info "Forces are not always symmetric with respect to the physical structure"
    We ensure that the returned forces are the derivatve of the obtained DFT
    energy with respect to positions *within the discretisation* encoded in
    the [`PlaneWaveBases`](@ref). Note, that as a result we are unable to make sure
    that forces always keep the symmetries of the physical structure, simply because
    the discretised problem (encoded in the `basis`) may be unable to represent numerically
    all physical symmetries, i.e. `basis.symmetries` may be only a subset
    of `model.symmetries`.

    For some use cases (e.g. geometry optimisations) it may be important to ensure
    that one keeps only force compoments that keep the symmetries of the stucture.
    in this case one an manually call `symmetrize_forces(forces_reduced; model.symmetries)`
    on the `forces_reduced` returned from this function to project out those force
    directions not keeping the symmetries of the phyiscal model.
"""
@timing function compute_forces(basis::PlaneWaveBasis{T}, ψ, occupation; kwargs...) where {T}
    # no explicit symmetrization is performed here, it is the
    # responsibility of each term to return symmetric forces
    forces_per_term = [compute_forces(term, basis, ψ, occupation; kwargs...)
                       for term in basis.terms]
    sum(filter(!isnothing, forces_per_term))
end

"""
Compute the Cartesian forces of an obtained SCF solution in Hartree / Bohr.
Returns a list of lists of forces
`[[force for atom in positions] for (element, positions) in atoms]`
which has the same structure as the `atoms` object passed to the underlying [`Model`](@ref).

!!! info "Forces are not always symmetric with respect to the physical structure"
    We ensure that the returned forces are the derivative of the obtained DFT
    energy with respect to positions within the discretisation encoded in
    the [`PlaneWaveBases`](@ref). As a result forces may not always be symmetric
    with respect to the physical structure. For more details, see the documentation
    of [`compute_forces`](@ref).
"""
function compute_forces_cart(basis::PlaneWaveBasis, ψ, occupation; kwargs...)
    forces_reduced = compute_forces(basis, ψ, occupation; kwargs...)
    covector_red_to_cart.(basis.model, forces_reduced)
end

function compute_forces(scfres)
    compute_forces(scfres.basis, scfres.ψ, scfres.occupation; scfres.ρ, scfres.τ)
end
function compute_forces_cart(scfres)
    compute_forces_cart(scfres.basis, scfres.ψ, scfres.occupation; scfres.ρ, scfres.τ)
end

# Internal function used to not only compute a force, but also symmetrise with respect
# to a custom set of symmetries. Currently only used in the DFTK calculator; if it becomes
# broadly useful, we should perhaps promote this to a proper API function.
function _compute_forces_cart_symmetrized(scfres; symmetries)
    forces_reduced = compute_forces(scfres)
    forces_reduced = symmetrize_forces(scfres.basis.model, forces_reduced; symmetries)
    covector_red_to_cart.(scfres.basis.model, forces_reduced)
end
