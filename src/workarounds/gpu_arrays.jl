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

"""
Specialised Magnetic term operator construction for for AbstractGPUArray
"""
function ene_ops(term::TermMagnetic, basis::PlaneWaveBasis{T}, ψ::Vector{A}, occupation;
                 kwargs...) where {T, A<:AbstractGPUArray}

    ops = [MagneticFieldOperator(basis, kpoint, term.Apotential)
           for (ik, kpoint) in enumerate(basis.kpoints)]
    if isnothing(ψ) || isnothing(occupation)
        return (; E=T(Inf), ops)
    end

    E = zero(T)
    for (ik, k) in enumerate(basis.kpoints)
        # Vectorised based on above for GPU compatibility
        # Not efficient due to intensive kernel launch with eachcol
        wt = basis.kweights[ik]
        # occ needs to be on CPU because eachcol + broadcasting makes an Array
        occ = to_cpu(occupation[ik])
        E += sum(real.(dot.(eachcol(ψ[ik]), Ref(ops[ik]) .* eachcol(ψ[ik]))) .* wt .* occ)
    end
    E = mpi_sum(E, basis.comm_kpts)

    (; E, ops)
end

