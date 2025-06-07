using LinearAlgebra
using GPUArraysCore
using Preferences

# https://github.com/JuliaGPU/CUDA.jl/issues/1565
LinearAlgebra.dot(x::AbstractGPUArray, D::Diagonal, y::AbstractGPUArray) = x' * (D * y)

for fun in (:potential_terms, :kernel_terms)
    @eval function DftFunctionals.$fun(fun::DispatchFunctional, ρ::AT,
                                       args...) where {AT <: AbstractGPUArray{Float64}}
        # Fallback implementation for the GPU: Transfer to the CPU and run computation there
        cpuify(::Nothing) = nothing
        cpuify(x::AbstractArray) = Array(x)
        $fun(fun, Array(ρ), cpuify.(args)...)
    end
end
