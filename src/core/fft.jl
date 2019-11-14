import FFTW
import FourierTransforms
import Primes

function is_fft_size_ok_for_generic(size::Integer)
    # TODO FourierTransforms has a bug, which is triggered
    #      only in some factorisations, see
    #      https://github.com/JuliaComputing/FourierTransforms.jl/issues/10
    # Everything is fine if we have up to one prime factor,
    # which is not two, also we want to avoid large primes
    penalty = 100
    sum((k == 2 ? 0 : (k > 7 ? penalty : v))
        for (k, v) in Primes.factor(size)) <= 1
end

function next_working_fft_size_for_generic(size)
    while !is_fft_size_ok_for_generic(size)
        size += 1
    end
    size
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
inv(p::GenericPlan{T}) where T = GenericPlan{T}(inv.(p.subplans), T(1 / p.factor))

function generic_plan_fft(data::AbstractArray{T, 3}) where T
    GenericPlan{T}([FourierTransforms.plan_fft(data[:, 1, 1]),
                 FourierTransforms.plan_fft(data[1, :, 1]),
                 FourierTransforms.plan_fft(data[1, 1, :])], T(1))
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

"""
Plan a FFT of type `T` and size `fft_size`, spending some time on finding an optimal algorithm.
Both an inplace and an out-of-place FFT plan are returned.
"""
function build_fft_plans(T, fft_size)
    tmp = Array{Complex{T}}(undef, fft_size...)
    if T == Float64
        ipFFT = FFTW.plan_fft!(tmp, flags=FFTW.MEASURE)
        opFFT = FFTW.plan_fft(tmp, flags=FFTW.MEASURE)
        return ipFFT, opFFT
    elseif T == Float32
        # TODO For Float32 there are issues with aligned FFTW plans.
        #      Using unaligned FFTW plans is discouraged, but we do it anyways
        #      here as a quick fix. We should reconsider this in favour of using
        #      a parallel wisdom anyways in the future.
        ipFFT = FFTW.plan_fft!(tmp, flags=FFTW.MEASURE | FFTW.UNALIGNED)
        opFFT = FFTW.plan_fft(tmp, flags=FFTW.MEASURE | FFTW.UNALIGNED)
        return ipFFT, opFFT
    end

    # Fall back to FourierTransforms
    # Note: FourierTransforms has no support for in-place FFTs at the moment
    # ... also it's extension to multi-dimensional arrays is broken and
    #     the algo only works for some cases
    @assert all(is_fft_size_ok_for_generic.(fft_size))

    # opFFT = FourierTransforms.plan_fft(tmp)   # TODO When multidim works
    opFFT = generic_plan_fft(tmp)               # Fallback for now
    # TODO Can be cut once FourierTransforms supports AbstractFFTs properly
    ipFFT = DummyInplace{typeof(opFFT)}(opFFT)

    ipFFT, opFFT
end
