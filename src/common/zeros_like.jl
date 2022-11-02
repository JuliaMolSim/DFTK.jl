# Create an array of same type as X filled with zeros, minimizing the number
# of allocations.
function zeros_like(X::AbstractArray, T::Type=eltype(X), dims::Integer...=size(X)...)
    Z = similar(X, T, dims...)
    Z .= 0
    Z
end
zeros_like(X::AbstractArray, dims::Integer...) = zeros_like(X, eltype(X), dims...)
zeros_like(X::Array, T::Type=eltype(X), dims::Integer...=size(X)...) = zeros(T, dims...)
zeros_like(X::StaticArray, T::Type=eltype(X), dims::Integer...=size(X)...) = @SArray zeros(T, dims...)

function copy_like(array_model::AbstractArray, src::AbstractArray)
    copy!(similar(array_model, eltype(src), size(src)...), src)
end

copy_like(array_model::Array, src::Array) = src

# function copy_like(array_model::Array, src::Array, T::Type=eltype(src), dims::Integer...=size(src)...)
#     T == eltype(src) && dims == size(src) && src
#     copy_like(array_model, src, T, dims...)
# end
