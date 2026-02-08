# Phonon symmetry operations
#
# This file implements the transformation of dynamical matrices under symmetry operations
# to enable computing phonons only in the irreducible Brillouin zone.
#
# Mathematical background:
# ========================
#
# A symmetry operation (W, w) transforms atomic positions as:
#   r' = W*r + w  (in reduced coordinates)
#
# For a displacement of atom s in direction α with wave vector q:
#   u_{s,α}(r) = e^{iq·r} e_α δ_{r,s}
#
# Under symmetry, this transforms to:
#   u'(r') = u(W^{-1}(r' - w)) = e^{iq·W^{-1}(r' - w)} e_α δ_{W^{-1}(r' - w), s}
#         = e^{i(W'^{-1}q)·r'} e^{-iq·W^{-1}w} e_α δ_{r', W*s + w}
#         = e^{i(S^{-1}q)·r'} e^{-iq·τ} e_α δ_{r', s'}
#
# where S = W' and τ = -W^{-1}w (reciprocal space symmetry parameters)
# and s' is the atom to which s is mapped by the symmetry.
#
# The dynamical matrix D^q_{s,α;t,β} relates the displacement of atom s in direction α
# to the force on atom t in direction β, both with wave vector q.
#
# Under a symmetry operation (W, w), we have:
#   D^{Sq}_{s',α';t',β'} = e^{iq·τ} W^{-T}_{α'α} D^q_{s,α;t,β} W^{-1}_{ββ'}
#
# where s' = symop(s), t' = symop(t), and the indices are transformed by the rotation.
# The phase factor e^{iq·τ} comes from the translation part of the symmetry.
#
# In reduced coordinates, the transformation is:
#   D^{Sq}_{s',α';t',β'} = e^{iq·τ} W^{-T}_{α'α} D^q_{s,α;t,β} W^{-1}_{ββ'}
#
# This allows us to:
# 1. Compute D^q for q in the irreducible BZ
# 2. Use symmetry to obtain D^{Sq} for any S that maps q to another point
# 3. Build the full dynamical matrix for all q-points from irreducible computations

using LinearAlgebra

"""
    apply_symop_dynmat(symop::SymOp, model::Model, dynmat, q)

Apply a symmetry operation to a dynamical matrix computed at q-point `q`.
Returns the dynamical matrix at the symmetry-transformed q-point `Sq = symop.S * q`.

The dynamical matrix is expected in the format `dynmat[α, s, β, t]` where
`α, β` are Cartesian directions (1:3) and `s, t` are atom indices.

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
    
    # The dynamical matrix transforms as a map from vectors to covectors
    # In reduced coordinates: D' = W^{-T} D W^{-1}
    # But we also need to account for:
    # 1. The atom permutation induced by the symmetry
    # 2. The phase factor from the translation part
    
    dynmat_symm = zero(dynmat)
    
    # Phase factor from translation: e^{iq·τ}
    phase = cis(2π * dot(q, τ))
    
    # For each pair of atoms in the transformed system
    for s_prime = 1:n_atoms
        for t_prime = 1:n_atoms
            # Find which atoms in the original system map to s_prime and t_prime
            s = find_symmetry_preimage(model.positions, model.positions[s_prime], 
                                      symop; tol_symmetry=SYMMETRY_TOLERANCE)
            t = find_symmetry_preimage(model.positions, model.positions[t_prime],
                                      symop; tol_symmetry=SYMMETRY_TOLERANCE)
            
            # Transform the matrix block: W^{-T} * D_{s,t} * W^{-1}
            # This transformation is in reduced coordinates
            W_inv = inv(W)
            W_inv_T = W_inv'
            
            # Apply transformation to the 3×3 block
            dynmat_symm[:, s_prime, :, t_prime] = phase * W_inv_T * dynmat[:, s, :, t] * W_inv
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
