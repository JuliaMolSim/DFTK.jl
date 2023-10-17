# Convert to Cartesian a dynamical matrix in reduced coordinates.
function dynmat_to_cart(model, dynamical_matrix)
    positions = model.positions
    n_atoms = length(positions)
    lattice = model.lattice
    inv_lattice = compute_inverse_lattice(lattice)

    cart_mat = zero.(dynamical_matrix)
    # The dynamical matrix `D` acts on vectors `dr` and gives a covector `dF`:
    #   dF = D · dr.
    # Thus the transformation between reduced and Cartesian coordinates is not a comatrix.
    # To transform `dynamical_matrix` from reduced coordinates to `cart_mat` in Cartesian
    # coordinates, we write
    #   dr_cart = lattice · dr_red,
    #   ⇒ dr_redᵀ · D_red · dr_red = dr_cartᵀ · lattice⁻ᵀ · D_red · lattice⁻¹ · dr_cart
    #                              = dr_cartᵀ · D_cart · dr_cart
    #   ⇒ D_cart = lattice⁻ᵀ · D_red · lattice⁻¹.
    for τ in 1:n_atoms
        for η in 1:n_atoms
            cart_mat[:, η, :, τ] = inv_lattice' * dynamical_matrix[:, η, :, τ] * inv_lattice
        end
    end
    cart_mat
end

# Compute the dynamical matrix in the form of a ``3×n_{\rm atoms}×3×n_{\rm atoms}`` tensor
# in reduced coordinates.
@timing function compute_dynmat(basis::PlaneWaveBasis, ψ, occupation; kwargs...)
    dynmats_per_term = [compute_dynmat(term, basis, ψ, occupation; kwargs...)
                        for term in basis.terms]
    sum(filter(!isnothing, dynmats_per_term))
end

# Cartesian form of [`compute_dynmat`](@ref).
function compute_dynmat_cart(basis::PlaneWaveBasis, ψ, occupation; kwargs...)
    dynmats_reduced = compute_dynmat(basis, ψ, occupation; kwargs...)
    dynmat_to_cart(basis.model, dynmats_reduced)
end
