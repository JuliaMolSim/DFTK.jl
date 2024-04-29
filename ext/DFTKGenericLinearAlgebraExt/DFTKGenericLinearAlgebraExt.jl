module DFTKGenericLinearAlgebraExt
using DFTK
using DFTK: DummyInplace
using LinearAlgebra
using AbstractFFTs
import AbstractFFTs: Plan, ScaledPlan,
                     fft, ifft, bfft, fft!, ifft!, bfft!,
                     plan_fft, plan_ifft, plan_bfft, plan_fft!, plan_ifft!, plan_bfft!,
                     rfft, irfft, brfft, plan_rfft, plan_irfft, plan_brfft,
                     fftshift, ifftshift,
                     rfft_output_size, brfft_output_size,
                     plan_inv, normalization
import Base: show, summary, size, ndims, length, eltype, *, inv, \
import LinearAlgebra: mul!

include("ctfft.jl")  # Main file of FourierTransforms.jl

# Utility functions to setup FFTs for DFTK. Most functions in here
# are needed to correct for the fact that FourierTransforms is not
# yet fully compliant with the AbstractFFTs interface and has still
# various bugs we work around.

function DFTK.next_working_fft_size(::Any, size::Integer)
    # TODO FourierTransforms has a bug, which is triggered
    #      only in some factorizations, see
    #      https://github.com/JuliaComputing/FourierTransforms.jl/issues/10
    nextpow(2, size)  # We fall back to powers of two to be safe
end
DFTK.default_primes(::Any) = (2, )

# Generic fallback function, Float32 and Float64 specialization in fft.jl
function DFTK.build_fft_plans!(tmp::AbstractArray{<:Complex})
    # Note: FourierTransforms has no support for in-place FFTs at the moment
    # ... also it's extension to multi-dimensional arrays is broken and
    #     the algo only works for some cases
    @assert all(ispow2, size(tmp))

    # opFFT = AbstractFFTs.plan_fft(tmp)   # TODO When multidim works
    # opBFFT = inv(opFFT).p
    opFFT  = generic_plan_fft(tmp)               # Fallback for now
    opBFFT = generic_plan_bfft(tmp)
    # TODO Can be cut once FourierTransforms supports AbstractFFTs properly
    ipFFT  = DummyInplace{typeof(opFFT)}(opFFT)
    ipBFFT = DummyInplace{typeof(opBFFT)}(opBFFT)

    ipFFT, opFFT, ipBFFT, opBFFT
end

struct GenericPlan{T}
    subplans
    factor::T
end

function Base.:*(p::GenericPlan, X::AbstractArray)
    pl1, pl2, pl3 = p.subplans
    ret = similar(X)
    for i = 1:size(X, 1), j = 1:size(X, 2)
        @views ret[i, j, :] .= pl3 * X[i, j, :]
    end
    for i = 1:size(X, 1), k = 1:size(X, 3)
        @views ret[i, :, k] .= pl2 * ret[i, :, k]
    end
    for j = 1:size(X, 2), k = 1:size(X, 3)
        @views ret[:, j, k] .= pl1 * ret[:, j, k]
    end
    p.factor .* ret
end

LinearAlgebra.mul!(Y, p::GenericPlan, X) = Y .= p * X
LinearAlgebra.ldiv!(Y, p::GenericPlan, X) = Y .= p \ X

length(p::GenericPlan) = prod(length, p.subplans)
*(p::GenericPlan{T}, fac::Number) where {T} = GenericPlan{T}(p.subplans, p.factor * T(fac))
*(fac::Number, p::GenericPlan{T}) where {T} = p * fac
\(p::GenericPlan, X) = inv(p) * X
inv(p::GenericPlan{T}) where {T} = GenericPlan{T}(inv.(p.subplans), 1 / p.factor)

function generic_plan_fft(data::AbstractArray{T, 3}) where {T}
    GenericPlan{T}([plan_fft(data[:, 1, 1]),
                    plan_fft(data[1, :, 1]),
                    plan_fft(data[1, 1, :])], T(1))
end
function generic_plan_bfft(data::AbstractArray{T, 3}) where {T}
    GenericPlan{T}([plan_bfft(data[:, 1, 1]),
                    plan_bfft(data[1, :, 1]),
                    plan_bfft(data[1, 1, :])], T(1))
end

end
