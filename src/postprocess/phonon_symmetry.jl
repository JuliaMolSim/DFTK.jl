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

The dynamical matrix is indexed as `dynmat[α, s, β, t]` where  
`(α, s)` is displacement component and atom, `(β, t)` is force component and atom.

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
    
    # The dynamical matrix in reduced coordinates transforms as:
    # D'^{Sq}[α', s', β', t'] = e^{iq·τ} * W_{α',α} * D^q[α, s, β, t] * W^{-T}_{β,β'}
    # where s' = W*s + w and t' = W*t + w
    
    dynmat_symm = zero(dynmat)
    
    # Phase factor from translation: e^{iq·τ}
    phase = cis(2π * dot(q, τ))
    
    W_inv_T = Matrix(inv(W)')
    
    # For each pair of atoms in the transformed system
    for s_prime = 1:n_atoms
        for t_prime = 1:n_atoms
            # Find which atoms in the original system map to s_prime and t_prime
            s = find_symmetry_preimage(model.positions, model.positions[s_prime], 
                                      symop; tol_symmetry=SYMMETRY_TOLERANCE)
            t = find_symmetry_preimage(model.positions, model.positions[t_prime],
                                      symop; tol_symmetry=SYMMETRY_TOLERANCE)
            
            # Apply transformation to the 3×3 block
            # D'[α', s', β', t'] = phase * W[α', α] * D[α, s, β, t] * W^{-T}[β, β']
            # This is equivalent to: W * D[:, s, :, t] * W^{-T}
            dynmat_symm[:, s_prime, :, t_prime] = phase * W * dynmat[:, s, :, t] * W_inv_T
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
"""
function find_irreducible_qpoint(q, symmetries)
    q_normalized = normalize_kpoint_coordinate(q)
    
    # For now, use a simple approach: return the first equivalent q-point
    # TODO: Could be optimized by using spglib's irreducible q-point mesh
    for symop in symmetries
        q_candidate = normalize_kpoint_coordinate(symop.S * q_normalized)
        # Check if this is the original q-point
        if norm(q_candidate - q_normalized) < SYMMETRY_TOLERANCE
            return (q_normalized, symop)
        end
    end
    
    # If we get here, q is already irreducible or no symmetry maps to it
    # Return q itself with identity operation
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
