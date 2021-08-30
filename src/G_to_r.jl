#
# Perform (i)FFTs.
#
# We perform two sets of (i)FFTs.

# For densities and potentials defined on the cubic basis set, r_to_G/G_to_r
# do a simple FFT/IFFT from the cubic basis set to the real-space grid.
# These function do not take a kpoint as input

# For orbitals, G_to_r converts the orbitals defined on a spherical
# basis set to the cubic basis set using zero padding, then performs
# an IFFT to get to the real-space grid. r_to_G performs an FFT, then
# restricts the output to the spherical basis set. These functions
# take a kpoint as input.

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

