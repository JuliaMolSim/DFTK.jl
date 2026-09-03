### GPU-specific implementations of DFTK's column-wise linear algebra helpers.
# The massive parallelism of the GPU can only be fully exploited when
# operating on whole arrays. For performance reasons, one should avoid
# explicitly looping over columns or elements. This approach is not
# necessarily the most performant on CPU, as the allocation of large
# temporary arrays hurts cache locality. It is also harder to read.
# (The analogous helpers used *inside* the eigensolver live in LOBPCGEigensolver.jl.)

using LinearAlgebra
using GPUArraysCore

function columnwise_dots(A::AbstractGPUArray{T}, B::AbstractGPUArray{T}) where {T}
    vec(sum(conj(A) .* B; dims=1))
end

function columnwise_dots(A::AbstractGPUArray{T}, M, B::AbstractGPUArray{T}) where {T}
    vec(sum(conj(A) .* (M * B); dims=1))
end

function columnwise_dots(A::AbstractGPUArray{T}, D::Diagonal, B::AbstractGPUArray{T}) where {T}
    vec(sum(conj(A) .* (D.diag .* B); dims=1))
end

function ldiv!(Y::AbstractGPUArray{T}, P::PreconditionerTPA, R::AbstractGPUArray{T}) where {T}
    if P.mean_kin === nothing
        ldiv!(Y, Diagonal(P.kin .+ P.default_shift), R)
    else
        Y .= (P.mean_kin' ./ (P.mean_kin' .+ P.kin)) .* R
    end
    Y
end

function mul!(Y::AbstractGPUArray{T}, P::PreconditionerTPA, R::AbstractGPUArray{T}) where {T}
    if P.mean_kin === nothing
        mul!(Y, Diagonal(P.kin .+ P.default_shift), R)
    else
        Y .= ((P.mean_kin' .+ P.kin) ./ P.mean_kin') .* R
    end
    Y
end
