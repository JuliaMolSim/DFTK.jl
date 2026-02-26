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
    dynmat_to_phonon_modes(basis::PlaneWaveBasis, dynmat)

Convert a dynamical matrix in reduced coordinates to phonon modes.

Returns the frequencies, mass matrix, and reduced and Cartesian eigenvectors and dynamical
matrices as a named tuple with fields: `mass_matrix`, `frequencies`, `dynmat`, `dynmat_cart`, 
`vectors`, `vectors_cart`.
"""
function dynmat_to_phonon_modes(basis::PlaneWaveBasis{T}, dynmat) where {T}
    # Convert dynamical matrix to Cartesian coordinates
    dynmat_cart = dynmat_red_to_cart(basis.model, dynmat)
    
    # Diagonalize in Cartesian coordinates
    n_atoms = length(basis.model.positions)
    M = reshape(mass_matrix(T, basis.model.atoms), 3*n_atoms, 3*n_atoms)

    res = eigen(reshape(dynmat_cart, 3*n_atoms, 3*n_atoms), M)
    maximum(abs, imag(res.values)) > sqrt(eps(T)) &&
        @warn "Some eigenvalues of the dynamical matrix have a large imaginary part."

    signs = sign.(real(res.values))
    frequencies = signs .* sqrt.(abs.(real(res.values)))
    vectors_cart = reshape(res.vectors, 3, n_atoms, 3, n_atoms)
    
    # Convert eigenvectors to reduced coordinates
    vectors = similar(vectors_cart)
    for s = 1:size(vectors, 2), t = 1:size(vectors, 4)
        vectors[:, s, :, t] = vector_cart_to_red(basis.model, vectors_cart[:, s, :, t])
    end

    (; mass_matrix=M, frequencies, dynmat, dynmat_cart, vectors, vectors_cart)
end

"""
Get phonon quantities. We return the frequencies, the mass matrix and reduced and Cartesian
eigenvectors and dynamical matrices.
"""
function phonon_modes(basis::PlaneWaveBasis{T}, ψ, occupation; kwargs...) where {T}
    dynmat = compute_dynmat(basis::PlaneWaveBasis, ψ, occupation; kwargs...)
    dynmat_to_phonon_modes(basis, dynmat)
end
function phonon_modes(scfres::NamedTuple; kwargs...)
    # TODO Pass down mixing and similar things to solve_ΩplusK_split
    phonon_modes(scfres.basis, scfres.ψ, scfres.occupation; scfres.ρ, scfres.ham,
                 scfres.occupation_threshold, scfres.εF, scfres.eigenvalues, kwargs...)
end

@doc raw"""
Compute the dynamical matrix in the form of a ``3×n_{\rm atoms}×3×n_{\rm atoms}`` tensor
in reduced coordinates.
"""
@timing function compute_dynmat(basis::PlaneWaveBasis{T}, ψ, occupation; q=zero(Vec3{T}),
                                ρ=nothing, ham=nothing, εF=nothing, eigenvalues=nothing,
                                kwargs...) where {T}
    @assert !basis.use_symmetries_for_kpoint_reduction
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


"""
Transform phonon modes under a symmetry operation.

Given phonon modes at q-point `q`, transform them to obtain modes at `Sq = S*q`
using symmetry operation `symop`.

The transformation works as follows:
- For each atom i, find the atom j such that positions[j] ≈ W*positions[i] + w (mod 1)
- The dynamical matrix transforms as: D'_jl = W D_ik W^{-1} * phase^2 where i->j, k->l

Here, W is the real-space rotation matrix and τ is the reciprocal translation
from the symmetry operation.
"""
function transform_phonon_modes(phonon_modes, symop::SymOp, model, q, basis)
    n_atoms = length(model.positions)
    W, w, S, τ = symop.W, symop.w, symop.S, symop.τ
    
    # Find the atom mapping: for each atom i, find which atom j it maps to under (W, w)
    atom_mapping = zeros(Int, n_atoms)
    for i = 1:n_atoms
        transformed_pos = W * model.positions[i] + w
        # Find which atom this maps to (modulo lattice vectors)
        for j = 1:n_atoms
            diff = transformed_pos - model.positions[j]
            if is_approx_integer(diff; atol=SYMMETRY_TOLERANCE)
                atom_mapping[i] = j
                break
            end
        end
        if atom_mapping[i] == 0
            error("Could not find mapping for atom $i under symmetry operation")
        end
    end
    
    # Transform the dynamical matrix in reduced coordinates
    # D'_jl = W D_ik W^{-1} where atom i maps to j and atom k maps to l
    dynmat = phonon_modes.dynmat
    W_inv = inv(W)
    dynmat_transformed = zero(dynmat)
    
    for i = 1:n_atoms, k = 1:n_atoms
        j = atom_mapping[i]
        l = atom_mapping[k]
        # Transform D[α, k, β, i] -> D'[α, l, β, j]
        for α = 1:3, β = 1:3
            dynmat_transformed[α, l, β, j] += sum(W[α, γ] * dynmat[γ, k, δ, i] * W_inv[δ, β] 
                                                    for γ = 1:3, δ = 1:3)
        end
    end
    
    # Use dynmat_to_phonon_modes to compute the modes from the transformed dynmat
    dynmat_to_phonon_modes(basis, dynmat_transformed)
end


"""
Compute phonon modes for all q-points in the Monkhorst-Pack grid using symmetries.

This function:
1. Builds an unfolded scfres (without BZ symmetry reduction)
2. Maps all q-points in the unfolded (reducible) BZ to the folded (irreducible) BZ
3. Computes phonon modes only for unique irreducible q-points
4. Transforms phonon modes using symmetries to obtain modes for all q-points
5. Returns a collection of phonon modes for all q-points in the unfolded BZ

The goal is that `all_phonon_modes(scfres)` ≈ `all_phonon_modes(unfold_bz(scfres))`.

Returns a vector of `(q=q_point, modes=phonon_modes_at_q)` named tuples, one for each
q-point in the Monkhorst-Pack grid (in the same order as the k-grid).
"""
function all_phonon_modes(scfres::NamedTuple; kwargs...)
    basis_orig = scfres.basis
    model = basis_orig.model
    
    # Build unfolded scfres if not already unfolded
    scfres_unfold = unfold_bz(scfres)
    basis_unfold = scfres_unfold.basis
    
    # Ensure the unfolded basis doesn't use symmetry for k-point reduction
    @assert !basis_unfold.use_symmetries_for_kpoint_reduction "Phonon calculations require an unfolded basis"
    
    # Get all q-points from the Monkhorst-Pack grid
    # We use the same grid as the k-grid
    kgrid = basis_orig.kgrid
    if !isa(kgrid, MonkhorstPack)
        error("all_phonon_modes currently only supports MonkhorstPack grids")
    end
    
    # Get all reducible q-points
    q_all = reducible_kcoords(kgrid).kcoords
    n_qpoints = length(q_all)
    
    # Get the irreducible q-points using the original basis (with symmetries)
    # Note: We need to work with the symmetry-reduced basis to find mappings
    basis_irred = basis_orig
    
    # Map each q-point to its irreducible representative
    q_to_irred = Dict{Int, Tuple{Int, SymOp}}()  # q_idx => (irred_idx, symop)
    q_irred_unique = Vector{Vec3{Float64}}()  # List of unique irreducible q-points
    q_irred_indices = Vector{Int}()  # Corresponding indices in q_all
    
    for (iq, q) in enumerate(q_all)
        # Check if this q-point is already in our irreducible list
        found = false
        for (iirred, q_irred) in enumerate(q_irred_unique)
            # Try to find a symmetry operation that maps q_irred to q
            for symop in basis_irred.symmetries
                Sq_irred = normalize_kpoint_coordinate(symop.S * q_irred)
                if isapprox(Sq_irred, q, atol=SYMMETRY_TOLERANCE)
                    q_to_irred[iq] = (iirred, symop)
                    found = true
                    break
                end
            end
            if found
                break
            end
        end
        
        if !found
            # This is a new irreducible q-point
            push!(q_irred_unique, q)
            push!(q_irred_indices, iq)
            identity_op = one(SymOp{Float64})
            q_to_irred[iq] = (length(q_irred_unique), identity_op)
        end
    end
    
    println("Total q-points: $n_qpoints")
    println("Irreducible q-points: $(length(q_irred_unique))")
    
    # Compute phonon modes for all unique irreducible q-points
    phonon_modes_irred = Vector{Any}(undef, length(q_irred_unique))
    for (iirred, q_irred) in enumerate(q_irred_unique)
        println("Computing phonon modes for q = $q_irred ($(iirred)/$(length(q_irred_unique)))")
        phonon_modes_irred[iirred] = phonon_modes(scfres_unfold; q=q_irred, kwargs...)
    end
    
    # Transform phonon modes to all q-points
    all_modes = Vector{Any}(undef, n_qpoints)
    for iq = 1:n_qpoints
        iirred, symop = q_to_irred[iq]
        q = q_all[iq]
        
        if isone(symop)
            # No transformation needed
            all_modes[iq] = (; q, modes=phonon_modes_irred[iirred])
        else
            # Transform the phonon modes
            q_irred = q_irred_unique[iirred]
            modes_transformed = transform_phonon_modes(phonon_modes_irred[iirred], 
                                                       symop, model, q_irred, basis_unfold)
            all_modes[iq] = (; q, modes=modes_transformed)
        end
    end
    
    all_modes
end
