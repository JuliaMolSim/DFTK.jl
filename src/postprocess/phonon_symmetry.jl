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
    find_irreducible_qpoint(q, symmetries)

Find a q-point in the irreducible Brillouin zone that is equivalent to `q` under symmetries,
along with the symmetry operation that maps the irreducible q to q.

# Arguments
- `q`: The q-point to reduce
- `symmetries`: Vector of symmetry operations

# Returns
- `(q_irred, symop)`: The irreducible q-point and the symmetry operation such that `symop.S * q_irred ≈ q`

# Note
This is a placeholder implementation. A proper implementation should find the lexicographically
smallest equivalent q-point to serve as the irreducible representative.
TODO: Implement proper irreducible q-point selection, possibly using spglib
"""
function find_irreducible_qpoint(q, symmetries)
    q_normalized = normalize_kpoint_coordinate(q)
    
    # Placeholder: Just return q itself with identity operation
    # A proper implementation would find the canonical (lexicographically smallest) 
    # representative among all symmetry-equivalent q-points
    return (q_normalized, SymOp(Mat3{Int}(I), Vec3(zeros(eltype(q), 3))))
end

"""
    get_irreducible_qpoints(qpoints, symmetries)

Get the irreducible set of q-points and the mapping from the full set to the irreducible set.

# Arguments  
- `qpoints`: Vector of all q-points
- `symmetries`: Vector of symmetry operations

# Returns
Named tuple with:
- `qpoints_irred`: Vector of irreducible q-points
- `mapping`: Vector of tuples (index_irred, symop) for each q-point in `qpoints`
"""
function get_irreducible_qpoints(qpoints, symmetries)
    # Group q-points by symmetry equivalence
    qpoints_normalized = normalize_kpoint_coordinate.(qpoints)
    qpoints_irred = eltype(qpoints)[]
    mapping = Vector{Tuple{Int, SymOp}}(undef, length(qpoints))
    
    for (iq, q) in enumerate(qpoints_normalized)
        # Check if q is equivalent to any already-found irreducible q-point
        found = false
        for (iq_irred, q_irred) in enumerate(qpoints_irred)
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
            # This is a new irreducible q-point
            push!(qpoints_irred, q)
            mapping[iq] = (length(qpoints_irred), SymOp(Mat3{Int}(I), Vec3(zeros(eltype(q), 3))))
        end
    end
    
    (; qpoints_irred, mapping)
end
