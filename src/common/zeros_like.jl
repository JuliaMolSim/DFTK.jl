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
