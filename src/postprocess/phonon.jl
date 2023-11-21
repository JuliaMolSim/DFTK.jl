# Convert to Cartesian a dynamical matrix in reduced coordinates.
function dynmat_red_to_cart(model::Model, dynamical_matrix)
    inv_lattice = model.inv_lattice

    # The dynamical matrix `D` acts on vectors `dr` and gives a covector `dF`:
    #   dF = D · dr.
    # Thus the transformation between reduced and Cartesian coordinates is not a comatrix.
    # To transform `dynamical_matrix` from reduced coordinates to `cart_mat` in Cartesian
    # coordinates, we write
    #   dr_cart = lattice · dr_red,
    #   ⇒ dr_redᵀ · D_red · dr_red = dr_cartᵀ · lattice⁻ᵀ · D_red · lattice⁻¹ · dr_cart
    #                              = dr_cartᵀ · D_cart · dr_cart
    #   ⇒ D_cart = lattice⁻ᵀ · D_red · lattice⁻¹.

    cart_mat = zero.(dynamical_matrix)
    for s = 1:size(cart_mat, 2), β = 1:size(cart_mat, 4)
        cart_mat[:, β, :, s] = inv_lattice' * dynamical_matrix[:, β, :, s] * inv_lattice
    end
    cart_mat
end

# Create a tensor equivalent to a``(n_{\rm atoms}×n_{\rm dim})^2`` diagonal matrix with the
# atomic masses of the atoms.
function get_masses(basis::PlaneWaveBasis{T}) where {T}
    model = basis.model
    positions = model.positions
    n_atoms = length(positions)

    atoms_mass = atomic_mass.(model.atoms)
    any(iszero.(atoms_mass)) && @error "Some elements have unknown masses"
    masses = zeros(T, 3, n_atoms, 3, n_atoms)
    for s in eachindex(atoms_mass)
        masses_s = @view masses[:, s, :, s]
        masses_s[diagind(masses_s)] .= atoms_mass[s]
    end
    masses
end

@doc raw"""
Solve the eigenproblem for a dynamical matrix. The phonon frequencies that are returned are
real numbers in ``{\rm cm}^{-1}``.
"""
function phonon_modes(basis::PlaneWaveBasis{T}, dynamical_matrix) where {T}
    model = basis.model
    positions = model.positions
    n_atoms = length(positions)

    mass_mat = reshape(get_masses(basis), 3*n_atoms, 3*n_atoms)
    cart_mat = reshape(dynmat_red_to_cart(model, dynamical_matrix), 3*n_atoms, 3*n_atoms)

    res = eigen(cart_mat, mass_mat)
    norm(imag(res.values)) > sqrt(eps(T)) && @warn "Some large imaginary phonon modes"

    signs = sign.(real(res.values))
    values = signs .* hartree_to_cm⁻¹ .* sqrt.(abs.(real(res.values)))

    (; values, res.vectors)
end
phonon_frequencies(args...; kwargs...) = phonon_modes(args...; kwargs...).values

"""
Compute the dynamical matrix in the form of a ``3×n_{\rm atoms}×3×n_{\rm atoms}`` tensor
in reduced coordinates.
"""
@timing function compute_dynmat(basis::PlaneWaveBasis{T}, ψ, occupation; q=zero(Vec3{T}),
                                ρ=nothing, ham=nothing, εF=nothing, eigenvalues=nothing,
                                kwargs...) where {T}
    is_perturbation_needed = any(t -> isa(t, TermAtomicLocal) ||
                                      isa(t, TermAtomicNonlocal), basis.terms)
    δψs = nothing
    δρs = nothing
    δoccupations = nothing
    if is_perturbation_needed
        model = basis.model
        positions = model.positions
        n_atoms = length(positions)
        n_dim = model.n_dim

        δHψs = compute_δHψ(basis, ψ; q)
        δρs = Array{Array{complex(eltype(basis)), 4}, 2}(undef, 3, n_atoms)
        for s = 1:n_atoms, α = 1:3
            δρs[α, s] = zeros(T, basis.fft_size..., basis.model.n_spin_components)
        end
        δoccupations = [zero.(occupation) for _ = 1:3, _ = 1:n_atoms]
        δψs = [zero.(ψ) for _ = 1:3, _ = 1:n_atoms]
        for s = 1:n_atoms, α = 1:n_dim
            (; δψ, δρ, δoccupation) = solve_ΩplusK_split(ham, ρ, ψ, occupation, εF,
                                                         eigenvalues, -δHψs[α, s]; q,
                                                         kwargs...)
            δoccupations[α, s] = δoccupation
            δρs[α, s] = δρ
            δψs[α, s] = δψ
        end
    end

    dynmats_per_term = [compute_dynmat(term, basis, ψ, occupation; ρ, δψs, δρs,
                                       δoccupations, q)
                        for term in basis.terms]
    sum(filter(!isnothing, dynmats_per_term))
end

"""
Cartesian form of [`compute_dynmat`](@ref).
"""
function compute_dynmat_cart(basis::PlaneWaveBasis, ψ, occupation; kwargs...)
    dynmats_reduced = compute_dynmat(basis, ψ, occupation; kwargs...)
    dynmat_red_to_cart(basis.model, dynmats_reduced)
end

function compute_dynmat(scfres::NamedTuple; kwargs...)
    compute_dynmat(scfres.basis, scfres.ψ, scfres.occupation; scfres.ρ, scfres.ham,
                   scfres.occupation_threshold, scfres.εF, scfres.eigenvalues, kwargs...)
end

function compute_dynmat_cart(scfres; kwargs...)
    compute_dynmat_cart(scfres.basis, scfres.ψ, scfres.occupation; scfres.ρ, scfres.ham,
                        scfres.occupation_threshold, scfres.εF, scfres.eigenvalues, kwargs...)
end

"""
Assemble the right-hand side term for the Sternheimer equation for all relevant quantities.
"""
@timing function compute_δHψ(basis::PlaneWaveBasis, ψ; kwargs...)
    δHψ_per_term = [compute_δHψ(term, basis, ψ; kwargs...) for term in basis.terms]
    sum(filter(!isnothing, δHψ_per_term))
end
