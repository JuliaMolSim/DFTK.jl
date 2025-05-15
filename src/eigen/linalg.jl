# Calculate the norms of the columns of an array
function columnwise_norms(X::AbstractArray{T}) where{T}
    vec(sqrt.(sum(abs2, X; dims=1)))
end

# Returns a vector of dot(A[:, i], B[:, i]), for all columns of A, B
@views function columnwise_dots(A::AbstractArray{T}, B::AbstractArray{T}) where {T}
    @assert size(A) == size(B)
    [real(dot(A[:, i], B[:, i])) for i = 1:size(A, 2)]
end

# Returns a vector of real(dot(A[:, i], M, B[:, i])), for all columns of
# A, B, and matrix M
@views function columnwise_dots(A::AbstractArray{T}, M, B::AbstractArray{T}) where {T}
    @assert size(A) == size(B)
    @assert size(M, 2) == size(B, 1)
    [real(dot(A[:, i], M, B[:, i])) for i = 1:size(A, 2)]
end