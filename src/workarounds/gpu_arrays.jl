using LinearAlgebra
using GPUArraysCore

# https://github.com/JuliaGPU/CUDA.jl/issues/1565
LinearAlgebra.dot(x::AbstractGPUArray, D::Diagonal, y::AbstractGPUArray) = x' * (D * y)

function lowpass_for_symmetry!(ρ::AT, basis;
    symmetries=basis.symmetries) where {AT <: AbstractGPUArray}
    all(isone, symmetries) && return ρ
    # lowpass_for_symmetry! currently uses scalar indexing, so we have to do this very ugly
    # thing for cases where ρ sits on a device (e.g. GPU)
    ρ_CPU = Array(ρ)
    ρ_CPU = lowpass_for_symmetry!(ρ_CPU, basis; symmetries)
    convert(AT, ρ_CPU)
end
