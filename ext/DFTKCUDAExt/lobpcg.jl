### GPU-specific implementations of functions called during LOBPCG
# The massive parallelism of the GPU can only be fully exploited when
# operating on whole arrays. For performance reasons, one should avoid
# explicitly looping over columns or elements. This approach is not
# necessarily the most performant on CPU, as the allocation of large
# temporary arrays hurts cache locality. It is also harder to read.

function DFTK.compute_λ(X::CUDA.CuArray{T}, AX::CUDA.CuArray{T}, BX::CUDA.CuArray{T}) where {T}
    num = sum(conj(X) .* AX, dims=1)
    den = sum(conj(X) .* BX, dims=1)
    vec(real.(num ./ den))
end

function DFTK.diag_prod(A::CUDA.CuArray{T}, B::CUDA.CuArray{T}) where {T}
    @assert size(A) == size(B)
    sum(conj(A) .* B; dims=1)
end

function DFTK.diag_prod(A::CUDA.CuArray{T}, M::CUDA.CuArray, B::CUDA.CuArray{T}) where {T}
    @assert size(A) == size(B)
    @assert size(M, 2) == size(B, 1)
    sum(conj(A) .* (M * B); dims=1)
end

function DFTK.diag_prod(A::CUDA.CuArray{T}, D::CUDA.Diagonal, B::CUDA.CuArray{T}) where {T}
    @assert size(A) == size(B)
    @assert length(D.diag) == size(B, 1)
    sum(conj(A) .* (D.diag .* B); dims=1)
end

function DFTK.ldiv!(Y::CUDA.CuArray{T}, P::PreconditionerTPA, R::CUDA.CuArray{T}) where {T}
    if P.mean_kin === nothing
        ldiv!(Y, Diagonal(P.kin .+ P.default_shift), R)
    else
        Y .= (P.mean_kin' ./ (P.mean_kin' .+ P.kin)) .* R
    end
    Y
end

function DFTK.mul!(Y::CUDA.CuArray{T}, P::PreconditionerTPA, R::CUDA.CuArray{T}) where {T}
    if P.mean_kin === nothing
        mul!(Y, Diagonal(P.kin .+ P.default_shift), R)
    else
        Y .= ((P.mean_kin' .+ P.kin) ./ P.mean_kin') .* R
    end
    Y
end