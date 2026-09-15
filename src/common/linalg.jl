# Calculate the norms of the columns of an array
function columnwise_norms(X::AbstractArray{T}) where{T}
    vec(sqrt.(sum(abs2, X; dims=1)))
end

# Returns a vector of dot(A[:, i], B[:, i]), for all columns of A, B. Overloaded on the GPU.
@views function columnwise_dots(A::AbstractArray{T}, B::AbstractArray{T}) where {T}
    [dot(A[:, i], B[:, i]) for i = 1:size(A, 2)]
end

# Returns a vector of dot(A[:, i], M, B[:, i]), for all columns of
# A, B, and matrix M. Overloaded on the GPU.
@views function columnwise_dots(A::AbstractArray{T}, M, B::AbstractArray{T}) where {T}
    [dot(A[:, i], M, B[:, i]) for i = 1:size(A, 2)]
end

"""
Compute a robust Cholesky factorization of a Hermitian matrix `A` by adding
a small diagonal shift if necessary to ensure positive definiteness.
"""
function shifted_cholesky(A::AbstractMatrix{T}) where {T}
    Treal = real(T)
    α = Treal(100)
    for _ = 1:5
        chol = cholesky(A; check=false)
        if issuccess(chol) && !any(isnan, chol.factors)
            return chol
        end

        # Factorization failed or produced NaNs; add regularization and retry.
        # Use @debug to avoid cluttering the standard output.
        @debug "Cholesky failed; adding regularization" α
        A += α * eps(Treal) * max(norm(A), one(Treal)) * I
        α *= 10
    end
    error("Cholesky factorization failed even with regularization.")
end
