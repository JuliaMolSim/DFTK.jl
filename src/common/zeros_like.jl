# Create an array of same type as X filled with zeros, minimizing the number
# of allocations.
function zeros_like(X::AbstractArray, T::Type=eltype(X), dims::Integer...=size(X)...)
    Z = similar(X, T, dims...)
    Z .= false
    Z
end
zeros_like(X::AbstractArray, dims::Integer...) = zeros_like(X, eltype(X), dims...)
zeros_like(X::Array, T::Type=eltype(X), dims::Integer...=size(X)...) = zeros(T, dims...)
zeros_like(X::StaticArray, T::Type=eltype(X), dims::Integer...=size(X)...) = @SArray zeros(T, dims...)

function convert_like(array_model::AbstractArray, src::AbstractArray)
    copy!(similar(array_model, eltype(src), size(src)...), src)
end
convert_like(array_model::Array, src::Array) = src
convert_like(array_model::Type,  src::AbstractArray) = convert(array_model, src)
