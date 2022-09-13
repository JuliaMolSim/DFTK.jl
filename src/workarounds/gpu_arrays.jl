# TODO: remove this when it is implemented in GPUArrays and CUDA
import LinearAlgebra.dot, LinearAlgebra.eigen
using LinearAlgebra
using GPUArrays
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
