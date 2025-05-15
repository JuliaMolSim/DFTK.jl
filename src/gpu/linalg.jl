### GPU-specific implementations of functions called during LOBPCG
# The massive parallelism of the GPU can only be fully exploited when
# operating on whole arrays. For performance reasons, one should avoid
# explicitly looping over columns or elements. This approach is not
# necessarily the most performant on CPU, as the allocation of large
# temporary arrays hurts cache locality. It is also harder to read.

using LinearAlgebra
using GPUArraysCore

function compute_λ(X::AbstractGPUArray{T}, AX::AbstractGPUArray{T}, BX::AbstractGPUArray{T}) where {T}
    num = sum(conj(X) .* AX, dims=1)
    den = sum(conj(X) .* BX, dims=1)
    vec(real.(num ./ den))
end

function columnwise_dots(A::AbstractGPUArray{T}, B::AbstractGPUArray{T}) where {T}
    @assert size(A) == size(B)
    sum(conj(A) .* B; dims=1)
end

function columnwise_dots(A::AbstractGPUArray{T}, M, B::AbstractGPUArray{T}) where {T}
    @assert size(A) == size(B)
    @assert size(M, 2) == size(B, 1)
    sum(conj(A) .* (M * B); dims=1)
end

function columnwise_dots(A::AbstractGPUArray{T}, D::Diagonal, B::AbstractGPUArray{T}) where {T}
    @assert size(A) == size(B)
    @assert length(D.diag) == size(B, 1)
    sum(conj(A) .* (D.diag .* B); dims=1)
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