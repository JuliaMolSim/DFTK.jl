import FFTW

#
# Perform (i)FFTs.
#
# We perform two sets of (i)FFTs.

# For densities and potentials defined on the cubic basis set, r_to_G/G_to_r
# do a simple FFT/IFFT from the cubic basis set to the real-space grid.
# These function do not take a k-point as input

# For orbitals, G_to_r converts the orbitals defined on a spherical
# basis set to the cubic basis set using zero padding, then performs
# an IFFT to get to the real-space grid. r_to_G performs an FFT, then
# restricts the output to the spherical basis set. These functions
# take a k-point as input.

"""
In-place version of `G_to_r`.
"""
@timing_seq function G_to_r!(f_real::AbstractArray3, basis::PlaneWaveBasis,
                             f_fourier::AbstractArray3)
    mul!(f_real, basis.opBFFT, f_fourier)
    f_real .*= basis.G_to_r_normalization
end
@timing_seq function G_to_r!(f_real::AbstractArray3, basis::PlaneWaveBasis,
                             kpt::Kpoint, f_fourier::AbstractVector; normalize=true)
    @assert length(f_fourier) == length(kpt.mapping)
    @assert size(f_real) == basis.fft_size

    # Pad the input data
    fill!(f_real, 0)
    f_real[kpt.mapping] = f_fourier

    # Perform an IFFT
    mul!(f_real, basis.ipBFFT, f_real)
    normalize && (f_real .*= basis.G_to_r_normalization)
    f_real
end

"""
    G_to_r(basis::PlaneWaveBasis, [kpt::Kpoint, ] f_fourier)

Perform an iFFT to obtain the quantity defined by `f_fourier` defined
on the k-dependent spherical basis set (if `kpt` is given) or the
k-independent cubic (if it is not) on the real-space grid.
"""
function G_to_r(basis::PlaneWaveBasis, f_fourier::AbstractArray; assume_real=true)
    # assume_real is true by default because this is the most common usage
    # (for densities & potentials)
    f_real = similar(f_fourier)
    @assert length(size(f_fourier)) ∈ (3, 4)
    # this exploits trailing index convention
    for iσ = 1:size(f_fourier, 4)
        @views G_to_r!(f_real[:, :, :, iσ], basis, f_fourier[:, :, :, iσ])
    end
    assume_real ? real(f_real) : f_real
end
function G_to_r(basis::PlaneWaveBasis, kpt::Kpoint, f_fourier::AbstractVector)
    G_to_r!(similar(f_fourier, basis.fft_size...), basis, kpt, f_fourier)
end


@doc raw"""
In-place version of `r_to_G!`.
NOTE: If `kpt` is given, not only `f_fourier` but also `f_real` is overwritten.
"""
@timing_seq function r_to_G!(f_fourier::AbstractArray3, basis::PlaneWaveBasis,
                             f_real::AbstractArray3)
    if isreal(f_real)
        f_real = complex.(f_real)
    end
    mul!(f_fourier, basis.opFFT, f_real)
    f_fourier .*= basis.r_to_G_normalization
end
@timing_seq function r_to_G!(f_fourier::AbstractVector, basis::PlaneWaveBasis,
                             kpt::Kpoint, f_real::AbstractArray3; normalize=true)
    @assert size(f_real) == basis.fft_size
    @assert length(f_fourier) == length(kpt.mapping)

    # FFT
    mul!(f_real, basis.ipFFT, f_real)

    # Truncate
    f_fourier .= view(f_real, kpt.mapping)
    normalize && (f_fourier .*= basis.r_to_G_normalization)
    f_fourier
end

"""
    r_to_G(basis::PlaneWaveBasis, [kpt::Kpoint, ] f_real)

Perform an FFT to obtain the Fourier representation of `f_real`. If
`kpt` is given, the coefficients are truncated to the k-dependent
spherical basis set.
"""
function r_to_G(basis::PlaneWaveBasis, f_real::AbstractArray)
    f_fourier = similar(f_real, complex(eltype(f_real)))
    @assert length(size(f_real)) ∈ (3, 4)
    # this exploits trailing index convention
    for iσ = 1:size(f_real, 4)
        @views r_to_G!(f_fourier[:, :, :, iσ], basis, f_real[:, :, :, iσ])
    end
    f_fourier
end
# TODO optimize this
function r_to_G(basis::PlaneWaveBasis, kpt::Kpoint, f_real::AbstractArray3)
    r_to_G!(similar(f_real, length(kpt.mapping)), basis, kpt, copy(f_real))
end

# returns matrix representations of the G_to_r and r_to_G matrices. For debug purposes.
function G_to_r_matrix(basis::PlaneWaveBasis{T}) where {T}
    ret = zeros(complex(T), prod(basis.fft_size), prod(basis.fft_size))
    for (iG, G) in enumerate(G_vectors(basis))
        for (ir, r) in enumerate(r_vectors(basis))
            ret[ir, iG] = cis(2π * dot(r, G)) / sqrt(basis.model.unit_cell_volume)
        end
    end
    ret
end
function r_to_G_matrix(basis::PlaneWaveBasis{T}) where {T}
    ret = zeros(complex(T), prod(basis.fft_size), prod(basis.fft_size))
    for (iG, G) in enumerate(G_vectors(basis))
        for (ir, r) in enumerate(r_vectors(basis))
            Ω = basis.model.unit_cell_volume
            ret[iG, ir] = cis(-2π * dot(r, G)) * sqrt(Ω) / prod(basis.fft_size)
        end
    end
    ret
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
function compute_fft_size(model::Model{T}, Ecut, kcoords=nothing;
                          ensure_smallprimes=true, algorithm=:fast, kwargs...) where T
    if algorithm == :fast
        Glims = compute_Glims_fast(model.lattice, Ecut; kwargs...)
    elseif algorithm == :precise
        @assert !isnothing(kcoords)

        # We build a temporary set of k-points here
        # We don't reuse this k-point construction for the pwbasis
        # because build_kpoints builds index mapping from the
        # k-point-specific basis to the global basis and thus the
        # fft_size needs to be final at k-point construction time
        Glims_temp    = compute_Glims_fast(model.lattice, Ecut; kwargs...)
        fft_size_temp = Tuple{Int, Int, Int}(2 .* Glims_temp .+ 1)
        kpoints_temp  = build_kpoints(model, fft_size_temp, kcoords, Ecut)

        Glims = compute_Glims_precise(model.lattice, Ecut, kpoints_temp; kwargs...)
    else
        error("Unknown fft_size_algorithm :$algorithm, try :fast or :precise")
    end

    # Optimize FFT grid size: Make sure the number factorises in small primes only
    fft_size = Vec3(2 .* Glims .+ 1)
    if ensure_smallprimes
        fft_size = nextprod.(Ref([2, 3, 5]), fft_size)
    end

    # TODO generic FFT is kind of broken for some fft sizes
    #      ... temporary workaround, see more details in workarounds/fft_generic.jl
    fft_size = next_working_fft_size(T, fft_size)
    Tuple{Int, Int, Int}(fft_size)
end


# This uses a more precise and slower algorithm than the one above,
# simply enumerating all G vectors and seeing where their difference
# is. It needs the kpoints to do so.
@timing function compute_Glims_precise(lattice::AbstractMatrix{T}, Ecut, kpoints; supersampling=2) where T
    recip_lattice  = compute_recip_lattice(lattice)
    recip_diameter = diameter(recip_lattice)
    Glims = [0, 0, 0]
    # get the bounding rectangle that contains all G-G' vectors
    # (and therefore densities and potentials)
    # This handles the case `supersampling=2`
    for kpt in kpoints
        # TODO Hack: kpt.G_vectors is an internal detail, better use G_vectors(basis, kpt)
        for G in kpt.G_vectors
            if norm(recip_lattice * (G + kpt.coordinate)) ≤ sqrt(2Ecut) - recip_diameter
                # each of the 8 neighbors (in ∞-norm) also belongs to the grid
                # so we can safely skip the search knowing at least one of them
                # will have larger |G-G′|.
                # Savings with this trick are surprisingly small :
                # for silicon, 50% at Ecut 30, 70% at Ecut 100
                continue
            end
            # TODO Hack: kpt.G_vectors is an internal detail, better use G_vectors(basis, kpt)
            for G′ in kpt.G_vectors
                for α = 1:3
                    @inbounds Glims[α] = max(Glims[α], abs(G[α] - G′[α]))
                end
            end
        end
    end
    if supersampling != 2
        # no guarantees here, we just do our best to satisfy the
        # target supersampling ratio
        Glims = round.(Int, supersampling ./ 2 .* Glims)
    end
    Glims
end

# Fast implementation, but sometimes larger than necessary.
function compute_Glims_fast(lattice::AbstractMatrix{T}, Ecut; supersampling=2, tol=sqrt(eps(T))) where T
    Gmax = supersampling * sqrt(2 * Ecut)
    bounding_rectangle(lattice, Gmax; tol=tol)
end


# returns the lengths of the bounding rectangle in reciprocal space
# that encloses the sphere of radius Gmax
function bounding_rectangle(lattice::AbstractMatrix{T}, Gmax; tol=sqrt(eps(T))) where {T}
    # If |B G| ≤ Gmax, then
    # |Gi| = |e_i^T B^-1 B G| ≤ |B^-T e_i| Gmax = |A_i| Gmax
    # with B the reciprocal lattice matrix, e_i the i-th canonical
    # basis vector and A_i the i-th column of the lattice matrix
    Glims = [norm(lattice[:, i]) / 2T(π) * Gmax for i in 1:3]

    # Round up, unless exactly zero (in which case keep it zero in
    # order to just have one G vector for 1D or 2D systems)
    Glims = [Glim == 0 ? 0 : ceil(Int, Glim .- tol) for Glim in Glims]
    Glims
end


# For Float32 there are issues with aligned FFTW plans, so we
# fall back to unaligned FFTW plans (which are generally discouraged).
_fftw_flags(::Type{Float32}) = FFTW.MEASURE | FFTW.UNALIGNED
_fftw_flags(::Type{Float64}) = FFTW.MEASURE

"""
Plan a FFT of type `T` and size `fft_size`, spending some time on finding an
optimal algorithm. (Inplace, out-of-place) x (forward, backward) FFT plans are returned.
"""
function build_fft_plans(T::Union{Type{Float32}, Type{Float64}}, fft_size)
    tmp = Array{Complex{T}}(undef, fft_size...)
    ipFFT = FFTW.plan_fft!(tmp, flags=_fftw_flags(T))
    opFFT = FFTW.plan_fft(tmp, flags=_fftw_flags(T))
    # backward by inverting and stripping off normalizations
    ipFFT, opFFT, inv(ipFFT).p, inv(opFFT).p
end


# TODO Some grid sizes are broken in the generic FFT implementation
# in FourierTransforms, for more details see workarounds/fft_generic.jl
# This function is needed to provide a noop fallback for grid adjustment for
# for floating-point types natively supported by FFTW
next_working_fft_size(::Type{Float32}, size::Int) = size
next_working_fft_size(::Type{Float64}, size::Int) = size
next_working_fft_size(T, sizes::Union{Tuple, AbstractArray}) = next_working_fft_size.(T, sizes)
