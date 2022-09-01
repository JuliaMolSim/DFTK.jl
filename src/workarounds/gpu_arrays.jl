#TODO: remove this when it is implemented in GPUArrays
import LinearAlgebra.dot, LinearAlgebra.eigen
using LinearAlgebra
using GPUArrays
using CUDA

LinearAlgebra.dot(x::AbstractGPUArray, D::Diagonal,y::AbstractGPUArray) = x'*(D*y)

function LinearAlgebra.eigen(A::Hermitian{T,AT}) where {T <: Complex,AT <: CuArray}
    vals, vects = CUDA.CUSOLVER.heevd!('V','U', A.data)
    (vectors = vects, values = vals)
end

function LinearAlgebra.eigen(A::Hermitian{T,AT}) where {T <: Real,AT <: CuArray}
    vals, vects = CUDA.CUSOLVER.syevd!('V','U', A.data)
    (vectors = vects, values = vals)
end
