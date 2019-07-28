"""
Structure containing a non-local potential in the Kleinman-Bylander form of projectors.
"""
struct PotNonLocal
    basis::PlaneWaveBasis

    # n_proj = ∑_atom ∑_l n_proj_per_l_for_atom * (2l + 1)
    proj_vectors::Vector  # n_k times (n_G × n_proj)
    proj_coeffs::Matrix   # n_proj × n_proj
end


function apply_fourier!(out, pot::PotNonLocal, ik::Int, in)
    P = pot.proj_vectors[ik]  # Projectors
    C = pot.proj_coeffs       # Coefficients
    out = P * (C * (P' * in))
end
