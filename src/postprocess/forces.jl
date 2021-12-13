# This uses the `compute_forces(term, ψ, occ; kwargs...)` function defined by all terms
"""
Compute the forces of an obtained SCF solution. Returns the forces wrt. the fractional
lattice vectors. To get cartesian forces use [`compute_forces_cart`](@ref).
Returns a list of lists of forces
`[[force for atom in positions] for (element, positions) in atoms]`
which has the same structure as the `atoms` object passed to the underlying [`Model`](@ref).
"""
@timing function compute_forces(basis::PlaneWaveBasis, ψ, occ; kwargs...)
    # TODO optimize allocs here
    T = eltype(basis)
    forces = [zeros(Vec3{T}, length(positions)) for (element, positions) in basis.model.atoms]
    for term in basis.terms
        f_term = compute_forces(term, basis, ψ, occ; kwargs...)
        if !isnothing(f_term)
            forces += f_term
        end
    end
    forces
end

"""
Compute the cartesian forces of an obtained SCF solution in Hartree / Bohr.
Returns a list of lists of forces
`[[force for atom in positions] for (element, positions) in atoms]`
which has the same structure as the `atoms` object passed to the underlying [`Model`](@ref).
"""
function compute_forces_cart(basis::PlaneWaveBasis, ψ, occ; kwargs...)
    lattice = basis.model.lattice
    forces = compute_forces(basis::PlaneWaveBasis, ψ, occ; kwargs...)
    [[lattice \ f for f in forces_for_element] for forces_for_element in forces]
end

function compute_forces(scfres)
    compute_forces(scfres.basis, scfres.ψ, scfres.occupation; ρ=scfres.ρ)
end
function compute_forces_cart(scfres)
    compute_forces_cart(scfres.basis, scfres.ψ, scfres.occupation; ρ=scfres.ρ)
end
