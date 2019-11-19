import FFTW
import Primes
include("FourierTransforms.jl/FourierTransforms.jl")

# Utility functions to setup FFTs for DFTK. Most functions in here
# are needed to correct for the fact that FourierTransforms is not
# yet fully compliant with the AbstractFFTs interface and has still
# various bugs we work around.

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
inv(p::GenericPlan{T}) where T = GenericPlan{T}(inv.(p.subplans), 1 / p.factor)

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


@doc raw"""
    determine_grid_size(lattice, Ecut; supersampling=2)

Determine the minimal grid size for the fourier grid ``C_ρ`` subject to the
kinetic energy cutoff `Ecut` for the wave function and a density  `supersampling` factor.
Optimise the grid afterwards for the FFT procedure by ensuring factorisation into
small primes.
The function will determine the smallest cube ``C_ρ`` containing the basis ``B_ρ``,
i.e. the wave vectors ``|G|^2/2 \leq E_\text{cut} ⋅ \text{supersampling}^2``.
For an exact representation of the density resulting from wave functions
represented in the basis ``B_k = \{G : |G + k|^2/2 \leq E_\text{cut}\}``,
`supersampling` should be at least `2`.
"""
function determine_grid_size(lattice::AbstractMatrix{T}, Ecut; supersampling=2, tol=1e-8, ensure_smallprimes=true) where T
    # See the documentation about the grids for details on the construction of C_ρ
    cutoff_Gsq = 2 * supersampling^2 * Ecut
    Gmax = [norm(lattice[:, i]) / 2T(π) * sqrt(cutoff_Gsq) for i in 1:3]
    # Round up, unless exactly zero (in which case keep it zero in
    # order to just have one G vector for 1D or 2D systems)
    for i = 1:3
        if Gmax[i] != 0
            Gmax[i] = ceil.(Int, Gmax[i] .- tol)
        end
    end

    # Optimise FFT grid size: Make sure the number factorises in small primes only
    if ensure_smallprimes
        Vec3([nextprod([2, 3, 5], 2gs + 1) for gs in Gmax])
    else
        Vec3([2gs+1 for gs in Gmax])
    end
end
function determine_grid_size(model::Model, Ecut; kwargs...)
    determine_grid_size(model.lattice, Ecut; kwargs...)
end


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
