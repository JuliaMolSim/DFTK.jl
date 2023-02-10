dalton_to_amu   = austrip(1u"u")
hartree_to_cm⁻¹ = ustrip(u"cm^-1", 1u"bohr^-1") ./ austrip(1u"c0") / 2π

# Create a ``(n_{\rm atoms}×n_{\rm dim})^2`` diagonal matrix with the atomic masses of the
# atoms.
function get_mass_matrix(basis)
    model = basis.model
    positions = model.positions
    n_atoms = length(positions)
    n_dim = model.n_dim

    charges = charge_nuclear.(model.atoms)
    # Some elements may be unknown (for example when using ElementGaussian).
    mass_matrix = zeros(ComplexF64, n_dim*n_atoms, n_dim*n_atoms)
    if !iszero(minimum(charges))
        for iτ in eachindex(charges)
            for γ in 1:n_dim
                mass_τγ = ustrip(periodic_table[charges[iτ]].atomic_mass) * dalton_to_amu
                mass_matrix[γ + (iτ - 1)*n_dim, γ + (iτ - 1)*n_dim] = mass_τγ
            end
        end
    else
        @warn "Some elements have unknown masses"
        mass_matrix[diagind(mass_matrix)] .= dalton_to_amu
    end

    mass_matrix
end

# Convert to Cartesian a dynamical matrix in reduced coordinates.
function dynmat_to_cart(basis, dynamical_matrix)
    model = basis.model
    positions = model.positions
    n_atoms = length(positions)
    lattice = model.lattice
    n_dim = model.n_dim
    inv_lattice = compute_inverse_lattice(lattice)[1:n_dim, 1:n_dim]

    cart_mat = zero.(dynamical_matrix)
    for i in 1:n_dim:n_dim*n_atoms
        for j in 1:n_dim:n_dim*n_atoms
            sub_matrix = dynamical_matrix[i:i+(n_dim-1), j:j+(n_dim-1)]
            cart_mat[i:i+(n_dim-1), j:j+(n_dim-1)] = inv_lattice' * sub_matrix * inv_lattice
        end
    end
    cart_mat
end

@doc raw"""
Obtain the phonons frequency in ``{\rm cm}^{-1}`` from a dynamical matrix.
"""
function phonon_eigenvalues(basis, dynamical_matrix)
    mass_matrix = get_mass_matrix(basis)
    cart_mat = dynmat_to_cart(basis, dynamical_matrix)

    eigenvalues = eigvals(cart_mat, mass_matrix)'
    norm(imag(eigenvalues)) > 1e-10 && @warn norm(imag(eigenvalues))
    signs = sign.(real.(eigenvalues))
    signs .* hartree_to_cm⁻¹ .* sqrt.(abs.(real.(eigenvalues)))
end
