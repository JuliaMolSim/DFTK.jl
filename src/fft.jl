import FFTW

@doc raw"""
Determine the minimal grid size for the cubic basis set to be able to
represent product of orbitals (with the default `supersampling=2`).

Optionally optimize the grid afterwards for the FFT procedure by
ensuring factorization into small primes.

The function will determine the smallest cube containing the wave vectors
 ``|G|^2/2 \leq E_\text{cut} ⋅ \text{supersampling}^2``.
For an exact representation of the density resulting from wave functions
represented in the spherical basis sets, `supersampling` should be at least `2`.
"""
function determine_grid_size(lattice::AbstractMatrix{T}, Ecut; supersampling=2, tol=1e-8,
                             ensure_smallprimes=true) where T
    cutoff_Gsq = 2 * supersampling^2 * Ecut
    Gmax = [norm(lattice[:, i]) / 2T(π) * sqrt(cutoff_Gsq) for i in 1:3]

    # Round up, unless exactly zero (in which case keep it zero in
    # order to just have one G vector for 1D or 2D systems)
    for i = 1:3
        if Gmax[i] != 0
            Gmax[i] = ceil.(Int, Gmax[i] .- tol)
        end
    end

    # Optimize FFT grid size: Make sure the number factorises in small primes only
    if ensure_smallprimes
        Vec3([nextprod([2, 3, 5], 2gs + 1) for gs in Gmax])
    else
        Vec3([2gs + 1 for gs in Gmax])
    end
end
function determine_grid_size(model::Model, Ecut; kwargs...)
    determine_grid_size(model.lattice, Ecut; kwargs...)
end

# For Float32 there are issues with aligned FFTW plans, so we
# fall back to unaligned FFTW plans (which are generally discouraged).
_fftw_flags(T::Float32) = FFTW.MEASURE | FFTW.UNALIGNED
_fftw_flags(T::Float64) = FFTW.MEASURE

"""
Plan a FFT of type `T` and size `fft_size`, spending some time on finding an
optimal algorithm. Both an inplace and an out-of-place FFT plan are returned.
"""
function build_fft_plans(T::Union{Type{Float32}, Type{Float64}}, fft_size)
    tmp = Array{Complex{T}}(undef, fft_size...)
    ipFFT = FFTW.plan_fft!(tmp, flags=_fftw_flags(T))
    opFFT = FFTW.plan_fft(tmp, flags=_fftw_flags(T))
    ipFFT, opFFT
end

# TODO Some grid sizes are broken in the generic FFT implementation
# in FourierTransforms, for more details see fft_generic.jl
# This function is needed to provide a noop fallback for grid adjustment for
# for floating-point types natively supported by FFTW
next_working_fft_size(::Type{Float32}, size) = size
next_working_fft_size(::Type{Float64}, size) = size
