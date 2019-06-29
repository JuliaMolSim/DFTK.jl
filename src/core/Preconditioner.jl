struct PreconditionerKinetic
    basis::PlaneWaveBasis
    α
end

"""Kinetic-energy based preconditioner.

Applies ``1 / (|k + G|^2 / 2 + α)`` to the vectors, when called with `ldiv!`.
This attempts to dampen the high-kinetic energy parts of the
Hamiltonian, thus making the Hamiltonian more well-conditioned.
"""
function PreconditionerKinetic(ham::Hamiltonian; α=0)
    PreconditionerKinetic(ham.basis, α)
end

import LinearAlgebra: ldiv!
function ldiv!(out, prec::PreconditionerKinetic, ik::Int, in)
    pw = prec.basis
    k = pw.kpoints[ik]

    qsq = [sum(abs2, pw.recip_lattice * (G + k)) for G in pw.basis_wf[ik]]
    diagonal = 1 ./ (qsq ./ 2 .+ prec.α)
    out .= diagonal .* in
end
