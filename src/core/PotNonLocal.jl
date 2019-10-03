# Data structures for representing the non-local potential in
# Kleinman-Bylander form and functionality to represent a k-Point block of it

include("PlaneWaveModel.jl")
using Memoize

"""
Structure containing a non-local potential in the Kleinman-Bylander form of projectors.
"""
struct PotNonLocal
    basis::PlaneWaveBasis

    # n_proj = ∑_atom ∑_l n_proj_per_l_for_atom * (2l + 1)
    # Projection coefficients and builder for projection vectors
    # The builder function is used to get the projection vectors for a particular
    # k-Point. Essentially the build_nonlocal of the model wrapped in some caching
    proj_coeffs   # (n_proj × n_proj)
    build_projection_vectors
end


"""
build_projection_vectors is expected to be a Function of the signature
(PlaneWaveBasis, Kpoint) -> (Matrix)
"""
function PotNonLocal(basis, proj_coeffs, build_projection_vectors)
    @memoize build_projectors_cached(basis, kpt) = build_projection_vectors(basis, kpt)
    PotNonLocal(basis, proj_coeffs, kpt -> build_projectors_cached(basis, kpt))
end


struct PotNonLocalBlock
    basis::PlaneWaveBasis
    kpt::Kpoint

    # Projection vectors and coefficients for this basis and k-Point
    # n_Gk = length(kpoint.basis)
    # n_proj = ∑_atom ∑_l n_proj_per_l_for_atom * (2l + 1)
    proj_vectors  # n_proj × n_proj
    proj_coeffs   # n_Gk × n_proj
end

import Base: *
function *(block::PotNonLocalBlock, X)
    P = block.proj_vectors
    C = block.proj_coeffs
    P * (C * (P' * X))
end

kblock(pot::PotNonLocal, kpt::Kpoint) =
    PotNonLocalBlock(pot.basis, kpt, pot.build_projection_vectors(kpt), pot.proj_coeffs)
