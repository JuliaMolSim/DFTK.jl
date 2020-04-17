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
if VERSION < v"1.4"
    # TODO Piracy, remove once we drop support for julia 1.3
    ldiv!(Y::AbstractVecOrMat, J::UniformScaling, B::AbstractVecOrMat) = (Y .= J.λ .\ B)
end


"""
(simplified version of) Tetter-Payne-Allan preconditioning
↑ M.P. Teter, M.C. Payne and D.C. Allan, Phys. Rev. B 40, 12255 (1989).
"""
mutable struct PreconditionerTPA{T <: Real}
    basis::PlaneWaveBasis
    kpt::Kpoint{T}
    kin::Vector{T}  # kinetic energy of every G
    mean_kin::Union{Nothing, Vector{T}}  # mean kinetic energy of every band
end

function PreconditionerTPA(basis::PlaneWaveBasis, kpt::Kpoint{T}) where T
    kin = Vector{T}([sum(abs2, basis.model.recip_lattice * (G + kpt.coordinate))
                     for G in G_vectors(kpt)] ./ 2)
    @assert basis.model.spin_polarisation in (:none, :collinear, :spinless)
    PreconditionerTPA{T}(basis, kpt, kin, nothing)
end

@views function ldiv!(Y, P::PreconditionerTPA, R)
    if P.mean_kin === nothing
        # This is arbitrary; the eigensolvers should support adaptive
        # preconditioning anyway.
        ldiv!(Y, Diagonal(P.kin .+ 1), R)
    else
        Threads.@threads for n = 1:size(Y, 2)
            Y[:, n] .= P.mean_kin[n] ./ (P.mean_kin[n] .+ P.kin) .* R[:, n]
        end
    end
    Y
end
ldiv!(P::PreconditionerTPA, R) = ldiv!(R, P, R)

# These are needed by eg direct minimization with CG
@views function mul!(Y, P::PreconditionerTPA, R)
    if P.mean_kin === nothing
        mul!(Y, Diagonal(P.kin .+ 1), R)
    else
        Threads.@threads for n = 1:size(Y, 2)
            Y[:, n] .= (P.mean_kin[n] .+ P.kin) ./ P.mean_kin[n] .* R[:, n]
        end
    end
    Y
end
mul!(P::PreconditionerTPA, R) = mul!(R, P, R)

function precondprep!(P::PreconditionerTPA, X)
    P.mean_kin = [real(dot(x, P.kin .* x)) for x in eachcol(X)]
end
