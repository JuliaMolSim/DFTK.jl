# TODO: remove this when it is implemented in GPUArrays and CUDA
import LinearAlgebra.dot, LinearAlgebra.eigen
using LinearAlgebra
using GPUArraysCore
using CUDA

# https://github.com/JuliaGPU/CUDA.jl/issues/1565
LinearAlgebra.dot(x::AbstractGPUArray, D::Diagonal,y::AbstractGPUArray) = x'*(D*y)

# https://github.com/JuliaGPU/CUDA.jl/issues/1572
function LinearAlgebra.eigen(A::Hermitian{T,AT}) where {T <: Complex,AT <: CuArray}
    vals, vects = CUDA.CUSOLVER.heevd!('V','U', A.data)
    (vectors = vects, values = vals)
end

function LinearAlgebra.eigen(A::Hermitian{T,AT}) where {T <: Real,AT <: CuArray}
    vals, vects = CUDA.CUSOLVER.syevd!('V','U', A.data)
    (vectors = vects, values = vals)
end

# Create an array of same type as X filled with zeros, minimizing the number
# of allocations.
function zeros_like(X, n, m)
    Z = similar(X, n, m)
    Z .= 0
    Z
end
