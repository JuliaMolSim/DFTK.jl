"""
Kinetic energy operator in a plane-wave basis.
"""
struct Kinetic
    basis::PlaneWaveBasis
end


function apply_fourier!(out_Xk, kinetic::Kinetic, ik, in_Xk)
    # Apply the Laplacian -Δ/2
    pw = kinetic.basis
    k = pw.kpoints[ik]

    qsq = [sum(abs2, pw.recip_lattice * (G + k)) for G in pw.wfctn_basis[ik]]
    out_Xk .= Diagonal(qsq / 2) * in_Xk
end
