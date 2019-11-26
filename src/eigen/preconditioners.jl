# Preconditioners
# TODO explain a bit of theory, or refer to a source

# A preconditioner is a type P that will get called as `Pk = P(ham, kpt)`
# and returns an object Pk that is expected to support
# `ldiv!(Y, Pk, R)` or `ldiv!(Pk, R)`
# Additionally if the solver supports adaptive preconditioning
# it will call `precondprep!(P,X)` right before calling `ldiv!`

import LinearAlgebra.ldiv!

precondprep!(P, X) = P  # This API is also used in Optim.jl

#
# Tetter-Payne-Allan preconditioning
#
"""
No preconditioning at all
"""
struct PreconditionerNone end
PreconditionerNone(ham::Hamiltonian, kpt::Kpoint) = LinearAlgebra.I

"""
(simplified version of) Tetter-Payne-Allan preconditioning
↑ M.P. Teter, M.C. Payne and D.C. Allan, Phys. Rev. B 40, 12255 (1989).
"""
mutable struct PreconditionerTPA{T <: Real}
    ham::Hamiltonian
    kpt::Kpoint{T}
    kin::Vector{T}  # kinetic energy of every G
    mean_kin::Union{Nothing, Vector{T}}  # mean kinetic energy of every band
end

function PreconditionerTPA(ham::Hamiltonian, kpt::Kpoint{T}) where T
    kin = Vector{T}([sum(abs2, ham.basis.model.recip_lattice * (G + kpt.coordinate))
                     for G in kpt.basis] ./ 2)
    @assert ham.basis.model.spin_polarisation in (:none, :collinear, :spinless)
    PreconditionerTPA{T}(ham, kpt, kin, nothing)
end

@views function ldiv!(Y, P::PreconditionerTPA, R)
    if P.mean_kin === nothing
        # This is arbitrary; the eigensolvers should support adaptive
        # preconditioning anyway.
        ldiv!(Y, Diagonal(P.kin .+ 1), R)
    else
        for n = 1:size(Y, 2)
            Y[:, n] .= P.mean_kin[n] ./ (P.mean_kin[n] .+ P.kin) .* R[:, n]
        end
    end
    Y
end
ldiv!(P::PreconditionerTPA, R) = ldiv!(R, P, R)

function precondprep!(P::PreconditionerTPA, X)
    P.mean_kin = [real(dot(x, P.kin .* x)) for x in eachcol(X)]
end
