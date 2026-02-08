# Convert to Cartesian a dynamical matrix in reduced coordinates.
function dynmat_red_to_cart(model::Model, dynmat)
    inv_lattice = model.inv_lattice

    # The dynamical matrix `D` acts on vectors `δr` and gives a covector `δF`:
    #   δF = D δr
    # We have δr_cart = lattice * δr_red, δF_cart = lattice⁻ᵀ δF_red, so
    #   δF_cart = lattice⁻ᵀ D_red lattice⁻¹ δr_cart
    dynmat_cart = zero.(dynmat)
    for s = 1:size(dynmat_cart, 2), α = 1:size(dynmat_cart, 4)
        dynmat_cart[:, α, :, s] = inv_lattice' * dynmat[:, α, :, s] * inv_lattice
    end
    dynmat_cart
end

# Create a ``3×n_{\rm atoms}×3×n_{\rm atoms}`` tensor equivalent to a diagonal matrix with
# the atomic masses of the atoms in a.u. on the diagonal.
function mass_matrix(T, atoms)
    n_atoms = length(atoms)
    atoms_mass = mass.(atoms)
    any(iszero.(atoms_mass)) && @warn "Some elements have unknown masses"
    masses = zeros(T, 3, n_atoms, 3, n_atoms)
    for s in eachindex(atoms_mass)
        masses[:, s, :, s] = austrip(atoms_mass[s]) * I(3)
    end
    masses
end
mass_matrix(model::Model{T}) where {T} = mass_matrix(T, model.atoms)

"""
Get phonon quantities. We return the frequencies, the mass matrix and reduced and Cartesian
eigenvectors and dynamical matrices.

If `qpoints` is provided, computes phonons for multiple q-points using symmetries to reduce
computation when possible. Returns results for all q-points.
"""
function phonon_modes(basis::PlaneWaveBasis{T}, ψ, occupation; q=nothing, qpoints=nothing, 
                      kwargs...) where {T}
    # Handle multiple q-points using symmetries
    if !isnothing(qpoints)
        if !isnothing(q)
            error("Cannot specify both q and qpoints")
        end
        # Build scfres-like named tuple for compute_phonons_on_grid
        scfres = (; basis, ψ, occupation, 
                   ρ=get(kwargs, :ρ, nothing),
                   ham=get(kwargs, :ham, nothing),
                   εF=get(kwargs, :εF, nothing),
                   eigenvalues=get(kwargs, :eigenvalues, nothing),
                   occupation_threshold=get(kwargs, :occupation_threshold, 1e-10))
        
        # Use compute_phonons_on_grid if qpoints is a MonkhorstPack or vector
        result = compute_phonons_on_grid(scfres, qpoints; kwargs...)
        
        # Convert dynmats to the format expected by phonon_modes
        n_atoms = length(basis.model.positions)
        M = reshape(mass_matrix(T, basis.model.atoms), 3*n_atoms, 3*n_atoms)
        
        # Process each q-point's dynamical matrix
        n_q = length(result.qcoords)
        all_frequencies = Vector{Vector{T}}(undef, n_q)
        all_vectors_cart = Vector{Array{Complex{T}, 4}}(undef, n_q)
        all_vectors = Vector{Array{Complex{T}, 4}}(undef, n_q)
        all_dynmat_cart = Vector{Array{Complex{T}, 4}}(undef, n_q)
        
        for iq = 1:n_q
            dynmat = result.dynmats[:, :, :, :, iq]
            dynmat_cart = dynmat_red_to_cart(basis.model, dynmat)
            modes_iq = _phonon_modes(basis, dynmat_cart)
            
            all_frequencies[iq] = modes_iq.frequencies
            all_vectors_cart[iq] = modes_iq.vectors_cart
            all_dynmat_cart[iq] = dynmat_cart
            
            vectors = similar(modes_iq.vectors_cart)
            for s = 1:size(vectors, 2), t = 1:size(vectors, 4)
                vectors[:, s, :, t] = vector_cart_to_red(basis.model, modes_iq.vectors_cart[:, s, :, t])
            end
            all_vectors[iq] = vectors
        end
        
        return (; mass_matrix=M, qcoords=result.qcoords, frequencies=all_frequencies,
                 dynmat=result.dynmats, dynmat_cart=all_dynmat_cart,
                 vectors=all_vectors, vectors_cart=all_vectors_cart,
                 qcoords_irred=result.qcoords_irred, n_irred=result.n_irred)
    end
    
    # Single q-point computation (original behavior)
    if isnothing(q)
        q = zero(Vec3{T})
    end
    
    dynmat = compute_dynmat(basis, ψ, occupation; q, kwargs...)
    dynmat_cart = dynmat_red_to_cart(basis.model, dynmat)

    modes = _phonon_modes(basis, dynmat_cart)
    vectors = similar(modes.vectors_cart)
    for s = 1:size(vectors, 2), t = 1:size(vectors, 4)
        vectors[:, s, :, t] = vector_cart_to_red(basis.model, modes.vectors_cart[:, s, :, t])
    end

    (; modes.mass_matrix, modes.frequencies, dynmat, dynmat_cart, vectors, modes.vectors_cart)
end
# Compute the frequencies and vectors. Internal because of the potential misuse:
# the diagonalization of the phonon modes has to be done in Cartesian coordinates.
function _phonon_modes(basis::PlaneWaveBasis{T}, dynmat_cart) where {T}
    n_atoms = length(basis.model.positions)
    M = reshape(mass_matrix(T, basis.model.atoms), 3*n_atoms, 3*n_atoms)

    res = eigen(reshape(dynmat_cart, 3*n_atoms, 3*n_atoms), M)
    maximum(abs, imag(res.values)) > sqrt(eps(T)) &&
        @warn "Some eigenvalues of the dynamical matrix have a large imaginary part."

    signs = sign.(real(res.values))
    frequencies = signs .* sqrt.(abs.(real(res.values)))

    (; mass_matrix=M, frequencies, vectors_cart=reshape(res.vectors, 3, n_atoms, 3, n_atoms))
end
function phonon_modes(scfres::NamedTuple; kwargs...)
    # TODO Pass down mixing and similar things to solve_ΩplusK_split
    phonon_modes(scfres.basis, scfres.ψ, scfres.occupation; scfres.ρ, scfres.ham,
                 scfres.occupation_threshold, scfres.εF, scfres.eigenvalues, kwargs...)
end

"""
Convenience wrapper to compute dynamical matrix from scfres.

# Arguments
- `scfres`: SCF results from `self_consistent_field`
- `q`: q-point at which to compute the dynamical matrix (default: Γ point)
- `kwargs...`: Additional keyword arguments passed to `compute_dynmat`

# Returns
- Dynamical matrix as a 3×n_atoms×3×n_atoms array in reduced coordinates
"""
function compute_dynmat(scfres::NamedTuple; kwargs...)
    compute_dynmat(scfres.basis, scfres.ψ, scfres.occupation; 
                   scfres.ρ, scfres.ham, scfres.εF, scfres.eigenvalues, 
                   scfres.occupation_threshold, kwargs...)
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
        # Get δH ψ
        δHψs_αs = compute_δHψ_αs(basis, ψ, α, s, q)
        isnothing(δHψs_αs) && continue
        # Response solver to get δψ
        (; δψ, δρ, δoccupation) = solve_ΩplusK_split(ham, ρ, ψ, occupation, εF, eigenvalues,
                                                     δHψs_αs; q, kwargs...)
        δoccupations[α, s] = δoccupation
        δρs[α, s] = δρ
        δψs[α, s] = δψ
    end
    # Query each energy term for their contribution to the dynamical matrix.
    dynmats_per_term = [compute_dynmat(term, basis, ψ, occupation; ρ, δψs, δρs,
                                       δoccupations, q)
                        for term in basis.terms]
    sum(filter(!isnothing, dynmats_per_term))
end

"""
Get ``δH·ψ``, with ``δH`` the perturbation of the Hamiltonian with respect to a position
displacement ``e^{iq·r}`` of the ``α`` coordinate of atom ``s``.
`δHψ[ik]` is ``δH·ψ_{k-q}``, expressed in `basis.kpoints[ik]`.
"""
@timing function compute_δHψ_αs(basis::PlaneWaveBasis, ψ, α, s, q)
    δHψ_per_term = [compute_δHψ_αs(term, basis, ψ, α, s, q) for term in basis.terms]
    filter!(!isnothing, δHψ_per_term)
    isempty(δHψ_per_term) && return nothing
    sum(δHψ_per_term)
end
