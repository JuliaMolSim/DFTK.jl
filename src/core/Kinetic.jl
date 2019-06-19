"""
Kinetic energy operator in a plane-wave basis.
"""
struct Kinetic
    basis::PlaneWaveBasis
end


function apply_fourier!(out, kinetic::Kinetic, ik, in)
    # Apply the Laplacian -Δ/2
    pw = kinetic.basis
    k = pw.kpoints[ik]

    qsq = [sum(abs2, pw.recip_lattice * (G + k)) for G in pw.basis_wf[ik]]
    out .= Diagonal(qsq / 2) * in
end
