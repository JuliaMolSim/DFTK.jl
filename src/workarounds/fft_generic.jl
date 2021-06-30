include("FourierTransforms.jl/FourierTransforms.jl")

# This is needed to flag that the fft_generic.jl file has already been loaded
const GENERIC_FFT_LOADED = true

if !isdefined(Main, :GenericLinearAlgebra)
    @warn("Code paths for generic floating-point types activated in DFTK. Remember to " *
          "add 'using GenericLinearAlgebra' to your user script. " *
          "See https://docs.dftk.org/stable/examples/arbitrary_floattype/ for details.")
end

# Utility functions to setup FFTs for DFTK. Most functions in here
# are needed to correct for the fact that FourierTransforms is not
# yet fully compliant with the AbstractFFTs interface and has still
# various bugs we work around.

function next_working_fft_size(::Any, size)
    # TODO FourierTransforms has a bug, which is triggered
    #      only in some factorizations, see
    #      https://github.com/JuliaComputing/FourierTransforms.jl/issues/10
    # To be safe we fall back to powers of two

    adjusted = nextpow(2, size)
    if adjusted != size
        @info "Changing fft size to $adjusted (smallest working size for generic FFTs)"
    end
    adjusted
end

# Generic fallback function, Float32 and Float64 specialization in fft.jl
function build_fft_plans(T, fft_size)
    tmp = Array{Complex{T}}(undef, fft_size...)

    # Note: FourierTransforms has no support for in-place FFTs at the moment
    # ... also it's extension to multi-dimensional arrays is broken and
    #     the algo only works for some cases
    @assert all(ispow2, fft_size)

    # opFFT = FourierTransforms.plan_fft(tmp)   # TODO When multidim works
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

function generic_apply(p::GenericPlan, X::AbstractArray)
    pl1, pl2, pl3 = p.subplans
    ret = similar(X)
    for i in 1:size(X, 1), j in 1:size(X, 2)
        @views ret[i, j, :] .= pl3 * X[i, j, :]
    end
    for i in 1:size(X, 1), k in 1:size(X, 3)
        @views ret[i, :, k] .= pl2 * ret[i, :, k]
    end
    for j in 1:size(X, 2), k in 1:size(X, 3)
        @views ret[:, j, k] .= pl1 * ret[:, j, k]
    end
    p.factor .* ret
end

LinearAlgebra.mul!(Y, p::GenericPlan, X) = Y .= p * X
LinearAlgebra.ldiv!(Y, p::GenericPlan, X) = Y .= p \ X

import Base: *, \, inv, length
length(p::GenericPlan) = prod(length, p.subplans)
*(p::GenericPlan, X::AbstractArray) = generic_apply(p, X)
*(p::GenericPlan{T}, fac::Number) where T = GenericPlan{T}(p.subplans, p.factor * T(fac))
*(fac::Number, p::GenericPlan{T}) where T = p * fac
\(p::GenericPlan, X) = inv(p) * X
inv(p::GenericPlan{T}) where T = GenericPlan{T}(inv.(p.subplans), 1 / p.factor)

function generic_plan_fft(data::AbstractArray{T, 3}) where T
    GenericPlan{T}([FourierTransforms.plan_fft(data[:, 1, 1]),
                    FourierTransforms.plan_fft(data[1, :, 1]),
                    FourierTransforms.plan_fft(data[1, 1, :])], T(1))
end
function generic_plan_bfft(data::AbstractArray{T, 3}) where T
    GenericPlan{T}([FourierTransforms.plan_bfft(data[:, 1, 1]),
                    FourierTransforms.plan_bfft(data[1, :, 1]),
                    FourierTransforms.plan_bfft(data[1, 1, :])], T(1))
end


# A dummy wrapper around an out-of-place FFT plan to make it appear in-place
# This is needed for some generic FFT implementations, which do not have in-place plans
struct DummyInplace{opFFT}
    fft::opFFT
end
LinearAlgebra.mul!(Y, p::DummyInplace, X) = (Y .= mul!(similar(X), p.fft, X))
LinearAlgebra.ldiv!(Y, p::DummyInplace, X) = (Y .= ldiv!(similar(X), p.fft, X))

import Base: *, \, length
*(p::DummyInplace, X) = p.fft * X
\(p::DummyInplace, X) = p.fft \ X
length(p::DummyInplace) = length(p.fft)
