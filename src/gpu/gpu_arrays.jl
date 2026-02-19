using LinearAlgebra
using GPUArraysCore
using Preferences

# https://github.com/JuliaGPU/CUDA.jl/issues/1565
LinearAlgebra.dot(x::AbstractGPUArray, D::Diagonal, y::AbstractGPUArray) = x' * (D * y)

# Norm of Hermitian matrices. See https://github.com/JuliaGPU/AMDGPU.jl/issues/843 for AMDGPU,
# and https://github.com/JuliaGPU/CUDA.jl/issues/2965 for CUDA.
function LinearAlgebra.norm(A::Hermitian{T, <:AbstractGPUArray}) where {T}
    upper_triangle = sum(abs2, triu(parent(A)))
    diago = sum(abs2, diag(parent(A)))
    sqrt(2upper_triangle - diago)
end

# Make sure that there is a CPU fallback for AbstractGPUArrays (e.g. for Duals)

@eval function DftFunctionals.potential_terms(fun::DispatchFunctional, ρ::AT,
                                              args...) where {AT <: AbstractGPUArray}
    # Fallback implementation for the GPU: Transfer to the CPU and run computation there
    cpuify(::Nothing) = nothing
    cpuify(x::AbstractArray) = Array(x)
    potential_terms(fun, Array(ρ), cpuify.(args)...)
end