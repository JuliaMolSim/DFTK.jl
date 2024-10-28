import FFTW
import AbstractFFTs: fft, fft!, ifft, ifft!

#
# Perform (i)FFTs.
#
# We perform two sets of (i)FFTs.

# For densities and potentials defined on the cubic basis set, fft/ifft
# do a simple FFT/IFFT from the cubic basis set to the real-space grid.
# These functions do not take a k-point as input

# For orbitals, ifft converts the orbitals defined on a spherical
# basis set to the cubic basis set using zero padding, then performs
# an IFFT to get to the real-space grid. fft performs an FFT, then
# restricts the output to the spherical basis set. These functions
# take a k-point as input.

"""
    G_vectors(fft_size::Tuple)

of given sizes.
"""
function G_vectors(fft_size::Union{Tuple,AbstractVector})
    # Note that a collect(G_vectors_generator(fft_size)) is 100-fold slower
    # than this implementation, hence the code duplication.
    start = .- cld.(fft_size .- 1, 2)
    stop  = fld.(fft_size .- 1, 2)
    axes  = [[collect(0:stop[i]); collect(start[i]:-1)] for i = 1:3]
    [Vec3{Int}(i, j, k) for i in axes[1], j in axes[2], k in axes[3]]
end

function G_vectors_generator(fft_size::Union{Tuple,AbstractVector})
    # The generator version is used mainly in symmetry.jl for lowpass_for_symmetry! and
    # accumulate_over_symmetries!, which are 100-fold slower with G_vector(fft_size).
    start = .- cld.(fft_size .- 1, 2)
    stop  = fld.(fft_size .- 1, 2)
    axes = [[collect(0:stop[i]); collect(start[i]:-1)] for i = 1:3]
    (Vec3{Int}(i, j, k) for i in axes[1], j in axes[2], k in axes[3])
end

"""
We define the FFTGrid struct, containing all the data required to perform FFTs. Namely:
- fft_size: defines the extent of the real and reciprocal space grids
- opFFT: out-of-place FFT plan
- ipFFT: in-place FFT plan
- opBFFT: out-of-place backward FFT plan
- ipBFFT: in-place backward FFT plan
- fft_normalization: normalization constant for FFTs
- ifft_normalization: normalization constant for backward FFTs
- G_vectors: grid coordinate in reciprocal space
- r_vectors: grid coordinate in real space
- architecture: information about the architecture on which FFT calculations take place
Note that the FFT plans are not normalized. Normalization takes place explicitely
when the fft()/ifft() functions are called
"""
struct FFTGrid{T, VT <: Real}
    fft_size::Tuple{Int, Int, Int}

    opFFT
    ipFFT
    opBFFT
    ipBFFT
    fft_normalization::T
    ifft_normalization::T

    G_vectors::Array{Vec3{Int}, 3}
    r_vectors::Array{Vec3{VT},  3}

    architecture::AbstractArchitecture
end

function FFTGrid(fft_size::Tuple{Int, Int, Int}, unit_cell_volume::T, 
                 arch::AbstractArchitecture) where T <: Real
    Gs = to_device(arch, G_vectors(fft_size))
    (ipFFT, opFFT, ipBFFT, opBFFT) = build_fft_plans!(similar(Gs, Complex{T}, fft_size))

    # Normalization constants
    # fft = fft_normalization * FFT
    # The convention we want is
    # ψ(r) = sum_G c_G e^iGr / sqrt(Ω)
    # so that the ifft has to normalized by 1/sqrt(Ω).
    # The other constant is chosen because FFT * BFFT = N
    ifft_normalization = 1/sqrt(unit_cell_volume)
    fft_normalization  = sqrt(unit_cell_volume) / length(ipFFT)

    VT = value_type(T)
    r_vectors = [(Vec3{VT}(idx.I) .- (1, 1, 1)) ./ VT.(fft_size)
                 for idx in CartesianIndices(fft_size)]
    r_vectors = to_device(arch, r_vectors)

    FFTGrid{T, VT}(fft_size, opFFT, ipFFT, opBFFT, ipBFFT, fft_normalization,
                   ifft_normalization, Gs, r_vectors, arch)
end

G_vectors(fft_grid::FFTGrid) = fft_grid.G_vectors
r_vectors(fft_grid::FFTGrid) = fft_grid.r_vectors

"""
In-place version of `ifft`.
"""
function ifft!(f_real::AbstractArray3, fft_grid::FFTGrid, f_fourier::AbstractArray3)
    mul!(f_real, fft_grid.opBFFT, f_fourier)
    f_real .*= fft_grid.ifft_normalization
end
function ifft!(f_real::AbstractArray3, fft_grid::FFTGrid,
               Gvec_mapping::Vector{Int}, f_fourier::AbstractVector; normalize=true)
    @assert length(f_fourier) == length(Gvec_mapping)
    @assert size(f_real) == fft_grid.fft_size

    # Pad the input data
    fill!(f_real, 0)
    f_real[Gvec_mapping] = f_fourier

    mul!(f_real, fft_grid.ipBFFT, f_real)  # perform IFFT
    normalize && (f_real .*= fft_grid.ifft_normalization)
    f_real
end

"""
    ifft(fft_grid::FFTGrid, [Gvec_mapping, ] f_fourier)

Perform an iFFT to obtain the quantity defined by `f_fourier` defined
on the k-dependent spherical basis set (if `Gvec_mapping` of a k-point is given)
or the k-independent cubic (if it is not) on the real-space grid.
"""
function ifft(fft_grid::FFTGrid, f_fourier::AbstractArray)
    f_real = similar(f_fourier)
    @assert length(size(f_fourier)) ∈ (3, 4)
    # this exploits trailing index convention
    for iσ = 1:size(f_fourier, 4)
        @views ifft!(f_real[:, :, :, iσ], fft_grid, f_fourier[:, :, :, iσ])
    end
    f_real
end
function ifft(fft_grid::FFTGrid, Gvec_mapping::Vector{Int}, f_fourier::AbstractVector; kwargs...)
    ifft!(similar(f_fourier, fft_grid.fft_size...), fft_grid, Gvec_mapping, f_fourier; kwargs...)
end
"""
Perform a real valued iFFT; see [`ifft`](@ref). Note that this function
silently drops the imaginary part.
"""
function irfft(fft_grid::FFTGrid{T}, f_fourier::AbstractArray) where {T}
    real(ifft(fft_grid, f_fourier))
end


@doc raw"""
In-place version of `fft!`.
NOTE: If `Gvec_mapping` is given, not only `f_fourier` but also `f_real` is overwritten.
"""
function fft!(f_fourier::AbstractArray3, fft_grid::FFTGrid, f_real::AbstractArray3)
    if eltype(f_real) <: Real
        f_real = complex.(f_real)
    end
    mul!(f_fourier, fft_grid.opFFT, f_real)
    f_fourier .*= fft_grid.fft_normalization
end
function fft!(f_fourier::AbstractVector, fft_grid::FFTGrid,
              Gvec_mapping::Vector{Int}, f_real::AbstractArray3; normalize=true)
    @assert size(f_real) == fft_grid.fft_size
    @assert length(f_fourier) == length(Gvec_mapping)

    # FFT
    mul!(f_real, fft_grid.ipFFT, f_real)

    # Truncate
    f_fourier .= view(f_real, Gvec_mapping)
    normalize && (f_fourier .*= fft_grid.fft_normalization)
    f_fourier
end

"""
    fft(fft_grid::FFTGrid, [Gvec_mapping, ] f_real)

Perform an FFT to obtain the Fourier representation of `f_real`. If
`Gvec_mapping` is given, the coefficients are truncated to the k-dependent
spherical basis set correspoding to a k-point G-vector mapping.
"""
function fft(fft_grid::FFTGrid{T}, f_real::AbstractArray{U}) where {T, U}
    f_fourier = similar(f_real, complex(promote_type(T, U)))
    @assert length(size(f_real)) ∈ (3, 4)
    for iσ = 1:size(f_real, 4)  # this exploits trailing index convention
        @views fft!(f_fourier[:, :, :, iσ], fft_grid, f_real[:, :, :, iσ])
    end
    f_fourier
end


# TODO optimize this
function fft(fft_grid::FFTGrid, Gvec_mapping::Vector{Int}, f_real::AbstractArray3; kwargs...)
    fft!(similar(f_real, length(Gvec_mapping)), fft_grid, Gvec_mapping, copy(f_real); kwargs...)
end

# returns matrix representations of the ifft and fft matrices. For debug purposes.
function ifft_matrix(fft_grid::FFTGrid{T}) where {T}
    ret = zeros(complex(T), prod(fft_grid.fft_size), prod(fft_grid.fft_size))
    for (iG, G) in enumerate(G_vectors(fft_grid))
        for (ir, r) in enumerate(r_vectors(fft_grid))
            ret[ir, iG] = cis2pi(dot(r, G)) * fft_grid.ifft_normalization
        end
    end
    ret
end
function fft_matrix(fft_grid::FFTGrid{T}) where {T}
    ret = zeros(complex(T), prod(fft_grid.fft_size), prod(fft_grid.fft_size))
    for (iG, G) in enumerate(G_vectors(fft_grid))
        for (ir, r) in enumerate(r_vectors(fft_grid))
            ret[iG, ir] = cis2pi(-dot(r, G)) * fft_grid.fft_normalization
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

If `factors` is not empty, ensure that the resulting fft_size contains all the factors
"""
function compute_fft_size(model::Model{T}, Ecut, kgrid=nothing;
                          ensure_smallprimes=true, algorithm=:fast, factors=1,
                          kwargs...) where {T}
    if algorithm == :fast
        Glims = compute_Glims_fast(model.lattice, Ecut; kwargs...)
    elseif algorithm == :precise
        @assert !isnothing(kgrid)
        kcoords = reducible_kcoords(kgrid).kcoords

        # We build a temporary set of k-points here
        # We don't reuse this k-point construction for the pwbasis
        # because build_kpoints builds index mapping from the
        # k-point-specific basis to the global basis and thus the
        # fft_size needs to be final at k-point construction time
        Glims_temp    = compute_Glims_fast(model.lattice, Ecut; kwargs...)
        fft_size_temp = Tuple{Int, Int, Int}(2 .* Glims_temp .+ 1)
        kpoints_temp  = build_kpoints(model, fft_size_temp, kcoords, Ecut;
                                      architecture=CPU())

        Glims = compute_Glims_precise(model.lattice, Ecut, kpoints_temp; kwargs...)
    else
        error("Unknown fft_size_algorithm :$algorithm, try :fast or :precise")
    end

    # TODO Make default small primes type-dependent, since generic FFT is broken for some
    #      prime factors ... temporary workaround, see more details in workarounds/fft_generic.jl
    if ensure_smallprimes
        smallprimes = default_primes(T)  # Usually (2, 3 ,5)
    else
        smallprimes = ()
    end

    # Consider only sizes that are (a) a product of small primes and (b) contain the factors
    fft_size = Vec3(2 .* Glims .+ 1)
    fft_size = next_compatible_fft_size(fft_size; factors, smallprimes)
    Tuple{Int, Int, Int}(fft_size)
end
function compute_fft_size(model::Model, Ecut::Quantity, kgrid=nothing; kwargs...)
    compute_fft_size(model, austrip(Ecut), kgrid; kwargs...)
end

"""
Find the next compatible FFT size
Sizes must (a) be a product of small primes only and (b) contain the factors.
If smallprimes is empty (a) is skipped.
"""
function next_compatible_fft_size(size::Int; smallprimes=(2, 3, 5), factors=(1, ))
    # This could be optimized
    is_product_of_primes(n) = isempty(smallprimes) || (n == nextprod(smallprimes, n))
    @assert all(is_product_of_primes, factors) # ensure compatibility between (a) and (b)
    has_factors(n) = rem(n, prod(factors)) == 0

    while !(has_factors(size) && is_product_of_primes(size))
        size += 1
    end
    size
end
function next_compatible_fft_size(sizes::Union{Tuple, AbstractArray}; kwargs...)
    next_compatible_fft_size.(sizes; kwargs...)
end

# This uses a more precise and slower algorithm than the one above,
# simply enumerating all G vectors and seeing where their difference
# is. It needs the kpoints to do so.
@timing function compute_Glims_precise(lattice::AbstractMatrix{T}, Ecut, kpoints;
                                       supersampling=2) where {T}
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
function compute_Glims_fast(lattice::AbstractMatrix{T}, Ecut;
                            supersampling=2, tol=sqrt(eps(T))) where {T}
    Gmax = supersampling * sqrt(2 * Ecut)
    recip_lattice = compute_recip_lattice(lattice)
    Glims = estimate_integer_lattice_bounds(recip_lattice, Gmax; tol)
    Glims
end

"""
Plan a FFT of type `T` and size `fft_size`, spending some time on finding an
optimal algorithm. (Inplace, out-of-place) x (forward, backward) FFT plans are returned.
"""
function build_fft_plans!(tmp::Array{Complex{Float64}})
    ipFFT = FFTW.plan_fft!(tmp; flags=FFTW.MEASURE)
    opFFT = FFTW.plan_fft(tmp;  flags=FFTW.MEASURE)
    # backwards-FFT by inverting and stripping off normalizations
    ipFFT, opFFT, inv(ipFFT).p, inv(opFFT).p
end
function build_fft_plans!(tmp::Array{Complex{Float32}})
    # For Float32 there are issues with aligned FFTW plans, so we
    # fall back to unaligned FFTW plans (which are generally discouraged).
    ipFFT = FFTW.plan_fft!(tmp; flags=FFTW.MEASURE | FFTW.UNALIGNED)
    opFFT = FFTW.plan_fft(tmp;  flags=FFTW.MEASURE | FFTW.UNALIGNED)
    # backwards-FFT by inverting and stripping off normalizations
    ipFFT, opFFT, inv(ipFFT).p, inv(opFFT).p
end
function build_fft_plans!(tmp::AbstractArray{Complex{T}}) where {T<:Union{Float32,Float64}}
    ipFFT = AbstractFFTs.plan_fft!(tmp)
    opFFT = AbstractFFTs.plan_fft(tmp)
    # backwards-FFT by inverting and stripping off normalizations
    ipFFT, opFFT, inv(ipFFT).p, inv(opFFT).p
end

# TODO Some grid sizes are broken in the generic FFT implementation
# in FourierTransforms, for more details see workarounds/fft_generic.jl
default_primes(::Type{Float32}) = (2, 3, 5)
default_primes(::Type{Float64}) = default_primes(Float32)
next_working_fft_size(::Type{Float32}, size::Int) = size
next_working_fft_size(::Type{Float64}, size::Int) = size
next_working_fft_size(T, sizes::Union{Tuple, AbstractArray}) = next_working_fft_size.(T, sizes)
