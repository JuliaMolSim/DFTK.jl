#TODO: remove this when it is implemented in GPUArrays
import LinearAlgebra.dot, LinearAlgebra.eigen, LinearAlgebra.RealHermSymComplexHerm
using LinearAlgebra
using GPUArrays

LinearAlgebra.dot(x::AbstractGPUArray, D::Diagonal,y::AbstractGPUArray) = x'*(D*y)

function LinearAlgebra.eigen(A::RealHermSymComplexHerm{T,AT}) where {T,AT <: CuArray}
    if eltype(A) <: Complex
        vals, vects = CUDA.CUSOLVER.heevd!('V','U', A.data)
    else
        vals, vects = CUDA.CUSOLVER.syevd!('V','U',A.data)
    end
    (vectors = vects, values = vals)
end
