"""
Create an array of same type as X filled with zeros, minimizing the number
of allocations. This unifies CPU and GPU code, as the output will always be on the
same device as the input.
"""
function zeros_like(X::AbstractArray, T::Type=eltype(X), dims::Integer...=size(X)...)
    Z = similar(X, T, dims...)
    Z .= zero(T)
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

function convert_like(array_model::AbstractArchitecture, src::AbstractArray)
    convert_like(get_array_type(array_model), src)
end
