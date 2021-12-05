# Preconditioners
# TODO explain a bit of theory, or refer to a source
# TODO homogenize with the terms interface

# A preconditioner is a type P that will get called as `Pk = P(basis, kpt)`
# and returns an object Pk that is expected to support
# ldiv! and mul! (both 2 and 3-arg versions)
# Additionally if the solver supports adaptive preconditioning
# it will call `precondprep!(P,X)` right before calling `ldiv!`

import LinearAlgebra: ldiv!
import LinearAlgebra: mul!

precondprep!(P, X) = P  # This API is also used in Optim.jl

"""
No preconditioning
"""
struct PreconditionerNone end
PreconditionerNone(basis, kpt) = I


"""
(simplified version of) Tetter-Payne-Allan preconditioning
↑ M.P. Teter, M.C. Payne and D.C. Allan, Phys. Rev. B 40, 12255 (1989).
"""
mutable struct PreconditionerTPA{T <: Real}
    basis::PlaneWaveBasis
    kpt::Kpoint
    kin::Vector{T}  # kinetic energy of every G
    mean_kin::Union{Nothing, Vector{T}}  # mean kinetic energy of every band
    default_shift::T # if mean_kin is not set by `precondprep!`, this will be used for the shift
end

function PreconditionerTPA(basis::PlaneWaveBasis{T}, kpt::Kpoint; default_shift=1) where T
    scaling = only([t for t in basis.model.term_types if t isa Kinetic]).scaling_factor
    kin = Vector{T}([scaling * sum(abs2, q) for q in Gplusk_vectors_cart(basis, kpt)] ./ 2)
    PreconditionerTPA{T}(basis, kpt, kin, nothing, default_shift)
end

@views function ldiv!(Y, P::PreconditionerTPA, R)
    if P.mean_kin === nothing
        ldiv!(Y, Diagonal(P.kin .+ P.default_shift), R)
    else
        Threads.@threads for n = 1:size(Y, 2)
            Y[:, n] .= P.mean_kin[n] ./ (P.mean_kin[n] .+ P.kin) .* R[:, n]
        end
    end
    Y
end
ldiv!(P::PreconditionerTPA, R) = ldiv!(R, P, R)
(Base.:\)(P::PreconditionerTPA, R) = ldiv!(P, copy(R))

# These are needed by eg direct minimization with CG
@views function mul!(Y, P::PreconditionerTPA, R)
    if P.mean_kin === nothing
        mul!(Y, Diagonal(P.kin .+ default_shift), R)
    else
        Threads.@threads for n = 1:size(Y, 2)
            Y[:, n] .= (P.mean_kin[n] .+ P.kin) ./ P.mean_kin[n] .* R[:, n]
        end
    end
    Y
end
(Base.:*)(P::PreconditionerTPA, R) = mul!(copy(R), P, R)

function precondprep!(P::PreconditionerTPA, X)
    P.mean_kin = [sum(real.(conj.(x) .* P.kin .* x)) for x in eachcol(X)]
end
