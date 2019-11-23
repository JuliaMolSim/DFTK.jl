# Preconditioners
# TODO explain a bit of theory, or refer to a source

# A preconditioner is a type P that will get called as `Pk = P(ham, kpt)`
# and returns an object Pk that is expected to support
# `ldiv!(Y, Pk, R)` or `Pk(Y, R)` (for the solvers that don't support adaptive preconditioning)
# or
# `Pk(Y, X, R)` (for those that do)

import LinearAlgebra.ldiv!

"""
No preconditioning at all
"""
struct PreconditionerNone end
PreconditionerNone(ham::Hamiltonian, kpt::Kpoint) = PreconditionerNone()
ldiv!(Y, P::PreconditionerNone, R) = Y .= R
(P::PreconditionerNone)(Y, R) = Y .= R
(P::PreconditionerNone)(Y, X, R) = Y .= R

"""
(simplified version of) Tetter-Payne-Allan preconditioning
↑ M.P. Teter, M.C. Payne and D.C. Allan, Phys. Rev. B 40, 12255 (1989).
"""
struct PreconditionerTPA{T <: Real}
    ham::Hamiltonian
    kpt::Kpoint{T}
    kin
end
function PreconditionerTPA(ham::Hamiltonian, kpt::Kpoint{T}) where T
    kin = Vector{T}([sum(abs2, ham.basis.model.recip_lattice * (G + kpt.coordinate))
                     for G in kpt.basis] ./ 2)
    @assert ham.basis.model.spin_polarisation in (:none, :collinear, :spinless)

    PreconditionerTPA{T}(ham, kpt, kin)
end

@views function (P::PreconditionerTPA)(Y, X, R)
    for n = 1:size(X, 2)
        mean_kin = real(dot(X[:, n], P.kin .* X[:, n]))
        Y[:, n] .= mean_kin ./ (mean_kin .+ P.kin) .* R[:, n]
    end
    Y
end
# This is kind of arbitrary; solvers should support adaptive preconditioning anyway
function (P::PreconditionerTPA)(Y, R)
    Y .= Diagonal(P.kin .+ 1) \ R
end
ldiv!(Y, P::PreconditionerTPA, R) = P(Y, R)
