using LinearAlgebra
using GPUArraysCore
using Preferences

# https://github.com/JuliaGPU/CUDA.jl/issues/1565
LinearAlgebra.dot(x::AbstractGPUArray, D::Diagonal, y::AbstractGPUArray) = x' * (D * y)

function lowpass_for_symmetry!(ρ::AbstractGPUArray, basis; symmetries=basis.symmetries)
    all(isone, symmetries) && return ρ
    # lowpass_for_symmetry! currently uses scalar indexing, so we have to do this very ugly
    # thing for cases where ρ sits on a device (e.g. GPU)
    ρ_CPU = lowpass_for_symmetry!(to_cpu(ρ), basis; symmetries)
    ρ .= to_device(basis.architecture, ρ_CPU)
end

for fun in (:potential_terms, :kernel_terms)
    @eval function DftFunctionals.$fun(fun::DispatchFunctional, ρ::AT,
                                       args...) where {AT <: AbstractGPUArray{Float64}}
        # Fallback implementation for the GPU: Transfer to the CPU and run computation there
        cpuify(::Nothing) = nothing
        cpuify(x::AbstractArray) = Array(x)
        $fun(fun, Array(ρ), cpuify.(args)...)
    end
end

# Note: This preference is temporary for now and can disappear at any time.
#       This is needed because for some reason I cannot arrive at the tests
#       to use CUDA version 11.8 instead of 12 (where Libxc is not supported)
function disable_libxc_cuda!()
    @set_preferences!("use_libxc_cuda" => "false")
end
