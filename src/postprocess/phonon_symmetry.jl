# Phonon symmetry operations
#
# This file implements the transformation of dynamical matrices under symmetry operations
# to enable computing phonons only in the irreducible Brillouin zone.
#
# Mathematical background:
# ========================
#
# A symmetry operation (W, w) in real space transforms atomic positions as:
#   r' = W*r + w  (in reduced coordinates)
#
# In reciprocal space, this induces the transformation:
#   S = W', τ = -W^{-1}*w
# such that q' = S*q and the phase transformation is e^{iq·τ}.
#
# The dynamical matrix D^q relates atomic displacement u to force F:
#   F_β(t) = D^q_β t, α s * u_α(s) * e^{iq·r_t}
#
# Under the symmetry operation:
# - Positions transform: s -> s' where r_s' = W*r_s + w
# - Displacements (vectors) in reduced coords: u'(s') = W * u(s)
# - Forces (covectors) in reduced coords: F'(t') = W^{-T} * F(t)  
# - q-point transforms: q' = S*q = W'*q
# - Phase: e^{iq'·τ} appears
#
# The dynamical matrix transformation in reduced coordinates becomes:
#   D^{q'}_{β t', α s'} = e^{iq·τ} * W^{-T}_{βγ} * D^q_{γ t, δ s} * W_{δα}
#
# where s' = W*s+w and t' = W*t+w are the images of atoms s and t under symmetry.

using LinearAlgebra

"""
    apply_symop_dynmat(symop::SymOp, model::Model, dynmat, q)

Apply a symmetry operation to a dynamical matrix computed at q-point `q`.
Returns the dynamical matrix at the symmetry-transformed q-point `Sq = symop.S * q`.

The dynamical matrix is indexed as `dynmat[β, t, α, s]` where  
`(α, s)` is displacement component and atom, `(β, t)` is force component and atom.
The relation is: F_β(t) = ∑_{α,s} dynmat[β, t, α, s] * u_α(s)

# Arguments
- `symop::SymOp`: The symmetry operation to apply
- `model::Model`: The model containing atomic positions and structure
- `dynmat`: The dynamical matrix at q-point `q` (3×n_atoms×3×n_atoms array)
- `q`: The q-point at which the dynamical matrix was computed

# Returns
- Transformed dynamical matrix at q-point `Sq`
"""
function apply_symop_dynmat(symop::SymOp, model::Model, dynmat, q)
    # If identity symmetry, no transformation needed
    isone(symop) && return dynmat
    
    n_atoms = length(model.positions)
    W = symop.W
    τ = symop.τ
    
    # Under symmetry W:
    # - Displacements (vectors): u'(s') = W * u(s) where s' = W*s + w
    # - Forces (covectors): F'(t') = W^{-T} * F(t) where t' = W*t + w
    #
    # Since F_β(t) = ∑_{α,s} D[β, t, α, s] * u_α(s), we have:
    # F'_β'(t') = W^{-T}_{β',β} * F_β(t)
    #           = W^{-T}_{β',β} * ∑_{α,s} D[β, t, α, s] * u_α(s)
    #           = W^{-T}_{β',β} * ∑_{α,s} D[β, t, α, s] * W^{-1}_{α,α'} * u'_α'(s')
    #
    # Therefore: D'[β', t', α', s'] = W^{-T}_{β',β} * D[β, t, α, s] * W^{-1}_{α,α'}
    
    dynmat_symm = zero(dynmat)
    
    # Phase factor from translation: e^{iq'·τ} where q' = S*q
    q_prime = symop.S * q
    phase = cis(2π * dot(q_prime, τ))
    
    W_inv = Matrix(inv(W))
    W_inv_T = W_inv'
    
    # For each pair of atoms in the transformed system
    for s_prime = 1:n_atoms
        for t_prime = 1:n_atoms
            # Find which atoms in the original system map to s_prime and t_prime
            s = find_symmetry_preimage(model.positions, model.positions[s_prime], 
                                      symop; tol_symmetry=SYMMETRY_TOLERANCE)
            t = find_symmetry_preimage(model.positions, model.positions[t_prime],
                                      symop; tol_symmetry=SYMMETRY_TOLERANCE)
            
            # Apply transformation to the 3×3 block
            # D'[β', t', α', s'] = phase * W^{-T}[β', β] * D[β, t, α, s] * W^{-1}[α, α']
            dynmat_symm[:, t_prime, :, s_prime] = phase * W_inv_T * dynmat[:, t, :, s] * W_inv
        end
    end
    
    dynmat_symm
end

"""
    compute_phonons_on_grid(scfres, qgrid::MonkhorstPack; kwargs...)
    
Compute phonons on a Monkhorst-Pack q-point grid, using symmetries to reduce computation.

Only computes dynamical matrices for irreducible q-points and unfolds to full grid using
symmetry operations. This provides speedup proportional to the number of symmetries.

# Arguments
- `scfres`: SCF results from `self_consistent_field`
- `qgrid`: Monkhorst-Pack grid specification for q-points
- `kwargs...`: Additional keyword arguments passed to `compute_dynmat`

# Returns
Named tuple with:
- `qcoords`: All q-point coordinates (full grid)
- `dynmats`: Dynamical matrices at all q-points (3×n_atoms×3×n_atoms×n_qpoints array)
- `qcoords_irred`: Irreducible q-point coordinates
- `n_irred`: Number of irreducible q-points computed
"""
function compute_phonons_on_grid(scfres, qgrid::MonkhorstPack; kwargs...)
    model = scfres.basis.model
    symmetries = scfres.basis.symmetries
    
    # Get all reducible q-points
    qcoords_full = reducible_kcoords(qgrid).kcoords
    
    # Get irreducible q-points using spglib (similar to k-point reduction)
    qdata_irred = irreducible_kcoords(qgrid, symmetries)
    qcoords_irred = qdata_irred.kcoords
    
    @info "Computing phonons: $(length(qcoords_irred)) irreducible q-points out of $(length(qcoords_full)) total"
    
    # Compute dynamical matrices for irreducible q-points
    n_atoms = length(model.positions)
    T = eltype(model.lattice)
    dynmats_irred = Vector{Array{Complex{T}, 4}}(undef, length(qcoords_irred))
    
    for (iq, q) in enumerate(qcoords_irred)
        dynmats_irred[iq] = compute_dynmat(scfres; q, kwargs...)
    end
    
    # Unfold to all q-points using symmetries
    dynmats_full = Vector{Array{Complex{T}, 4}}(undef, length(qcoords_full))
    
    for (iq_full, q_full) in enumerate(qcoords_full)
        # Find which irreducible q-point and symmetry to use
        found = false
        for (iq_irred, q_irred) in enumerate(qcoords_irred)
            for symop in symmetries
                q_symm = normalize_kpoint_coordinate(symop.S * q_irred)
                if norm(q_symm - q_full) < SYMMETRY_TOLERANCE
                    # Transform dynamical matrix from q_irred to q_full
                    dynmats_full[iq_full] = apply_symop_dynmat(symop, model, 
                                                               dynmats_irred[iq_irred], q_irred)
                    found = true
                    break
                end
            end
            found && break
        end
        @assert found "Could not map q-point $q_full to any irreducible q-point"
    end
    
    # Stack into a single array
    dynmats = cat(dynmats_full..., dims=5)
    
    (; qcoords=qcoords_full, dynmats, qcoords_irred, n_irred=length(qcoords_irred))
end

"""
    compute_phonons_on_grid(scfres, qcoords::AbstractVector; kwargs...)
    
Compute phonons on explicit q-points, using symmetries to reduce computation.

# Arguments
- `scfres`: SCF results from `self_consistent_field`
- `qcoords`: Vector of q-point coordinates  
- `kwargs...`: Additional keyword arguments passed to `compute_dynmat`

# Returns
Named tuple with:
- `qcoords`: All input q-point coordinates
- `dynmats`: Dynamical matrices at all q-points (3×n_atoms×3×n_atoms×n_qpoints array)
- `qcoords_irred`: Irreducible q-point coordinates that were computed
- `n_irred`: Number of irreducible q-points computed
"""
function compute_phonons_on_grid(scfres, qcoords::AbstractVector; kwargs...)
    model = scfres.basis.model
    symmetries = scfres.basis.symmetries
    
    # Normalize q-points
    qcoords_normalized = normalize_kpoint_coordinate.(qcoords)
    
    # Find irreducible set by grouping symmetry-equivalent q-points
    qcoords_irred = eltype(qcoords)[]
    mapping = Vector{Tuple{Int, SymOp}}(undef, length(qcoords))
    
    for (iq, q) in enumerate(qcoords_normalized)
        found = false
        for (iq_irred, q_irred) in enumerate(qcoords_irred)
            for symop in symmetries
                q_symm = normalize_kpoint_coordinate(symop.S * q_irred)
                if norm(q_symm - q) < SYMMETRY_TOLERANCE
                    mapping[iq] = (iq_irred, symop)
                    found = true
                    break
                end
            end
            found && break
        end
        
        if !found
            push!(qcoords_irred, q)
            mapping[iq] = (length(qcoords_irred), SymOp(Mat3{Int}(I), Vec3(zeros(eltype(q), 3))))
        end
    end
    
    @info "Computing phonons: $(length(qcoords_irred)) irreducible q-points out of $(length(qcoords)) total"
    
    # Compute dynamical matrices for irreducible q-points
    n_atoms = length(model.positions)
    T = eltype(model.lattice)
    dynmats_irred = Vector{Array{Complex{T}, 4}}(undef, length(qcoords_irred))
    
    for (iq, q) in enumerate(qcoords_irred)
        dynmats_irred[iq] = compute_dynmat(scfres; q, kwargs...)
    end
    
    # Unfold to all q-points using stored mapping
    dynmats_full = Vector{Array{Complex{T}, 4}}(undef, length(qcoords))
    
    for (iq, (iq_irred, symop)) in enumerate(mapping)
        q_irred = qcoords_irred[iq_irred]
        dynmats_full[iq] = apply_symop_dynmat(symop, model, dynmats_irred[iq_irred], q_irred)
    end
    
    # Stack into a single array
    dynmats = cat(dynmats_full..., dims=5)
    
    (; qcoords=qcoords_normalized, dynmats, qcoords_irred, n_irred=length(qcoords_irred))
end
