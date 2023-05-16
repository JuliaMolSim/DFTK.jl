using LinearAlgebra
using GPUArraysCore

# https://github.com/JuliaGPU/CUDA.jl/issues/1565
LinearAlgebra.dot(x::AbstractGPUArray, D::Diagonal, y::AbstractGPUArray) = x' * (D * y)

function lowpass_for_symmetry!(ρ::AbstractGPUArray, basis; symmetries=basis.symmetries)
    all(isone, symmetries) && return ρ
    # lowpass_for_symmetry! currently uses scalar indexing, so we have to do this very ugly
    # thing for cases where ρ sits on a device (e.g. GPU)
    ρ_CPU = lowpass_for_symmetry!(to_cpu(ρ), basis; symmetries)
    ρ .= to_device(basis.architecture, ρ_CPU)
end
