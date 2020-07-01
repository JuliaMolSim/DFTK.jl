import FFTW

# returns the lengths of the bounding rectangle in reciprocal space
# that encloses the sphere of radius Gmax
function bounding_rectangle(lattice::AbstractMatrix{T}, Gmax; tol=1e-8) where {T}
    # If |B G| ≤ Gmax, then
    # |Gi| ≤ |e_i^T B^-1 B G| ≤ |B^-T e_i| Gmax = |A_i| Gmax
    # with B the reciprocal lattice matrix and A_i the i-th column of the lattice matrix
    Glims_ = [norm(lattice[:, i]) / 2T(π) * Gmax for i in 1:3]

    # Round up, unless exactly zero (in which case keep it zero in
    # order to just have one G vector for 1D or 2D systems)
    Glims = [0, 0, 0]
    for i = 1:3
        if Glims_[i] != 0
            Glims[i] = ceil(Int, Glims_[i] .- tol)
        end
    end
    Glims
end

@doc raw"""
Determine the minimal grid size for the cubic basis set to be able to
represent product of orbitals (with the default `supersampling=2`).

Optionally optimize the grid afterwards for the FFT procedure by
ensuring factorization into small primes.

The function will determine the smallest parallelepiped containing the wave vectors
 ``|G|^2/2 \leq E_\text{cut} ⋅ \text{supersampling}^2``.
For an exact representation of the density resulting from wave functions
represented in the spherical basis sets, `supersampling` should be at least `2`.
"""
function determine_grid_size(lattice::AbstractMatrix{T}, Ecut; supersampling=2, tol=1e-8,
                             ensure_smallprimes=true) where T
    Gmax = supersampling * sqrt(2 * Ecut)
    Glims = bounding_rectangle(lattice, Gmax; tol=tol)

    # Optimize FFT grid size: Make sure the number factorises in small primes only
    if ensure_smallprimes
        Vec3([nextprod([2, 3, 5], 2gs + 1) for gs in Glims])
    else
        Vec3([2gs + 1 for gs in Glims])
    end
end
function determine_grid_size(model::Model, Ecut; kwargs...)
    determine_grid_size(model.lattice, Ecut; kwargs...)
end

# For Float32 there are issues with aligned FFTW plans, so we
# fall back to unaligned FFTW plans (which are generally discouraged).
_fftw_flags(T::Type{Float32}) = FFTW.MEASURE | FFTW.UNALIGNED
_fftw_flags(T::Type{Float64}) = FFTW.MEASURE

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
