# Create a ``(n_{\rm atoms}×n_{\rm dim})^2`` diagonal matrix with the atomic masses of the
# atoms.
function get_mass_matrix(basis)
    model = basis.model
    positions = model.positions
    n_atoms = length(positions)
    n_dim = model.n_dim

    charges = atomic_mass.(model.atoms)
    any(iszero.(charges)) && @error "Some elements have unknown masses"
    # Some elements may be unknown (for example when using ElementGaussian).
    mass_matrix = Diagonal(zeros(eltype(basis), n_dim*n_atoms, n_dim*n_atoms))
    for iτ in eachindex(charges)
        for γ in 1:n_dim
            mass_matrix[γ + (iτ - 1)*n_dim, γ + (iτ - 1)*n_dim] = charges[iτ]
        end
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
    # The dynamical matrix acts on vectors `dr` and gives a covector `dF`. Thus the
    # transformation between reduced and Cartesian coordinates is not a comatrix.
    # To transform `dynamical_matrix` from reduced coordinates to `cart_mat` in Cartesian
    # coordinates, we write
    #   dr_cart = lattice · dr_red
    #   dr_cartᵀ · dF_cart · dr_cart = dr_redᵀ · latticeᵀ · dF_cart · lattice · dr_red
    #                                = dr_redᵀ · dF_red · dr_red
    #   ⇒ dF_cart = latticeᵀ · dF_red · lattice
    #   ⇒ dF_red = lattice⁻ᵀ · dF_cart · lattice⁻¹
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

    # Is equivalent to `eigvals(cart_mat, Matrix(mass_matrix))` with `mass_matrix` diagonal.
    ismass_matrix = inv(sqrt(mass_matrix))
    eigenvalues = eigvals(ismass_matrix * cart_mat * ismass_matrix)'
    norm(imag(eigenvalues)) > 1e-10 && @warn norm(imag(eigenvalues))
    signs = sign.(real.(eigenvalues))
    signs .* hartree_to_cm⁻¹ .* sqrt.(abs.(real.(eigenvalues)))
end

function compute_δρ(scfres::NamedTuple; q=zero(Vec3{eltype(scfres.basis)}),
                    tol=1e-15, verbose=false)
    basis = scfres.basis
    model = basis.model
    positions = model.positions
    n_atoms = length(positions)
    n_dim = model.n_dim

    δHψs = compute_δHψ(scfres; q)
    δρs = Array{Array{complex(eltype(scfres.basis)), 4}, 2}(undef, n_dim, n_atoms)
    δoccupations = [zero.(scfres.occupation) for _ in 1:n_dim, _ in 1:n_atoms]
    δψs = [zero.(scfres.ψ) for _ in 1:n_dim, _ in 1:n_atoms]
    for τ in 1:n_atoms
        for γ in 1:n_dim
            δψ, δρ, δoccupation = solve_ΩplusK_split(scfres, -δHψs[γ, τ]; q=q, tol, verbose)
            δoccupations[γ, τ] = δoccupation
            δρs[γ, τ] = δρ
            δψs[γ, τ] = δψ
        end
    end
    (; δρs, δψs, δoccupations)
end
