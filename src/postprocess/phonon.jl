# Convert to Cartesian a dynamical matrix in reduced coordinates.
function dynmat_red_to_cart(model::Model, dynmat)
    inv_lattice = model.inv_lattice

    # The dynamical matrix `D` acts on vectors `dr` and gives a covector `dF`:
    #   dF = D · dr.
    # Thus the transformation between reduced and Cartesian coordinates is not a comatrix.
    # To transform `dynamical_matrix` from reduced coordinates to `dynmat_cart` in Cartesian
    # coordinates, we write
    #   dr_cart = lattice · dr_red,
    #   ⇒ dr_redᵀ · D_red · dr_red = dr_cartᵀ · lattice⁻ᵀ · D_red · lattice⁻¹ · dr_cart
    #                              = dr_cartᵀ · D_cart · dr_cart
    #   ⇒ D_cart = lattice⁻ᵀ · D_red · lattice⁻¹.

    dynmat_cart = zero.(dynmat)
    for s = 1:size(dynmat_cart, 2), α = 1:size(dynmat_cart, 4)
        dynmat_cart[:, α, :, s] = inv_lattice' * dynmat[:, α, :, s] * inv_lattice
    end
    dynmat_cart
end

# Create a ``3×n_{\rm atoms}×3×n_{\rm atoms}`` tensor (for consistency with the format of
# dynamical matrices) equivalent to a diagonal matrix with the atomic masses of the atoms in
# a.u. on the diagonal.
function mass_matrix(T, atoms)
    n_atoms = length(atoms)
    atoms_mass = atomic_mass.(atoms)
    any(iszero.(atoms_mass)) && @warn "Some elements have unknown masses"
    masses = zeros(T, 3, n_atoms, 3, n_atoms)
    for s in eachindex(atoms_mass)
        masses[:, s, :, s] = austrip(atoms_mass[s]) * I(3)
    end
    masses
end
mass_matrix(model::Model{T}) where {T} = mass_matrix(T, model.atoms)

@doc raw"""
Solve the eigenproblem for a dynamical matrix: returns the `frequencies` and eigenvectors
(`vectors`).
"""
function phonon_modes(basis::PlaneWaveBasis{T}, dynmat) where {T}
    n_atoms = length(basis.model.positions)
    M = reshape(mass_matrix(T, basis.model.atoms), 3*n_atoms, 3*n_atoms)

    res = eigen(reshape(dynmat, 3*n_atoms, 3*n_atoms), M)
    maximum(abs, imag(res.values)) > sqrt(eps(T)) &&
        @warn "Some eigenvalues of the dynamical matrix have a large imaginary part"

    signs = sign.(real(res.values))
    frequencies = signs .* sqrt.(abs.(real(res.values)))

    (; frequencies, res.vectors)
end
function phonon_modes_cart(basis::PlaneWaveBasis{T}, dynmat) where {T}
    dynmat_cart = dynmat_red_to_cart(basis.model, dynmat)
    phonon_modes(basis, dynmat_cart)
end

@doc raw"""
Compute the dynamical matrix in the form of a ``3×n_{\rm atoms}×3×n_{\rm atoms}`` tensor
in reduced coordinates.
"""
@timing function compute_dynmat(basis::PlaneWaveBasis{T}, ψ, occupation; q=zero(Vec3{T}),
                                ρ=nothing, ham=nothing, εF=nothing, eigenvalues=nothing,
                                kwargs...) where {T}
    n_atoms = length(basis.model.positions)
    δρs = [zeros(complex(T), basis.fft_size..., basis.model.n_spin_components)
           for _ = 1:3, _ = 1:n_atoms]
    δoccupations = [zero.(occupation) for _ = 1:3, _ = 1:n_atoms]
    δψs = [zero.(ψ) for _ = 1:3, _ = 1:n_atoms]
    for s = 1:n_atoms, α = 1:basis.model.n_dim
        δHψs_αs = compute_δHψ_αs(basis, ψ, α, s, q)
        isnothing(δHψs_αs) && continue
        (; δψ, δρ, δoccupation) = solve_ΩplusK_split(ham, ρ, ψ, occupation, εF, eigenvalues,
                                                     -δHψs_αs; q, kwargs...)
        δoccupations[α, s] = δoccupation
        δρs[α, s] = δρ
        δψs[α, s] = δψ
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

# TODO: Document relation to non-local potential in future phonon PR.
"""
Assemble the right-hand side term for the Sternheimer equation for all relevant quantities:
Compute the perturbation of the Hamiltonian with respect to a variation of the local
potential produced by a displacement of the atom s in the direction α.
"""
@timing function compute_δHψ_αs(basis::PlaneWaveBasis, ψ, α, s, q)
    δHψ_per_term = [compute_δHψ_αs(term, basis, ψ, α, s, q) for term in basis.terms]
    filter!(!isnothing, δHψ_per_term)
    isempty(δHψ_per_term) && return nothing
    sum(δHψ_per_term)
end
