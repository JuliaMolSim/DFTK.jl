#TODO: remove this when it is implemented in GPUArrays
import LinearAlgebra.dot
using LinearAlgebra
using GPUArrays
import Base.iszero, Base.isone

LinearAlgebra.dot(x::AbstractGPUArray, D::Diagonal,y::AbstractGPUArray) = x'*(D*y)

Base.iszero(x::AbstractGPUMatrix{T}) where {T} = all(iszero, x)

function Base.isone(x::AbstractGPUMatrix{T}) where {T}
    n,m = size(x)
    m != n && return false
    all(iszero, x-I)
end
