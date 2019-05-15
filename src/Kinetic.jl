"""
Kinetic energy operator in a plane-wave basis.
"""
struct Kinetic
    basis::PlaneWaveBasis
end


function apply_fourier!(out_Xk, kinetic::Kinetic, ik, in_Xk)
    # Apply the laplacian -Δ/2
    pw = kinetic.basis
    out_Xk .= Diagonal(pw.qsq[ik] / 2) * in_Xk
end
