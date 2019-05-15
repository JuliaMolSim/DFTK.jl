using FFTW

abstract type AbstractBasis end

struct PlaneWaveBasis{T <: Real} <: AbstractBasis
    #
    # Lattice info used for construction
    #
    """3x3 real-space lattice vectors, in columns."""
    lattice::Matrix{T}

    """3x3 reciprocal-space lattice vectors, in columns."""
    recip_lattice::Matrix{T}

    """Volume of the unit cell"""
    unit_cell_volume::T

    #
    # Plane-Wave mesh
    #
    """The kpoints of the basis"""
    kpoints::Vector{Vector{T}}

    """The weights associated to the above kpoints"""
    kweights::Vector{T}

    """Wave vectors of the plane wave basis functions in reciprocal space"""
    Gs::Vector{Vector{T}}

    """Maximal kinetic energy |G + k|^2 in Hartree"""
    Ecut::T

    """Index of the DC fourier component in the plane-wave mesh"""
    idx_DC::Int

    """
    Masks (list of indices), which select the plane waves required for each
    kpoint, such that the resulting plane-wave basis reaches the
    selected Ecut threshold.
    """
    kmask::Vector{Vector{Int}}

    """
    Cache for the expression |G + k|^2 for each wave vector corresponding
    to a k-Point. The length is the number of k-Points and each individual
    vector has length `length(kmask[ik])` where ik is the index into
    the k-Point.
    """
    qsq::Vector{Vector{T}}

    #
    # FFT and real-space grid Y*
    #
    """
    Coordinates of the grid points on the real-space grid Y*
    """
    grid_Yst::Array{Vector{T}, 3}

    """
    Translation table to transform from the
    plane-wave basis to the indexing convention used during the FFT
    (DC component in the middle, others periodically, asymmetrically around it
    """
    idx_to_fft::Vector{Vector{Int}}

    # Space to store planned FFT operators from FFTW
    """Plan for forward FFT"""
    FFT  # TODO maybe concrete types should be put here?

    """Plan for inverse FFT"""
    iFFT # TODO maybe concrete types should be put here?
end


Base.eltype(basis::PlaneWaveBasis{T}) where T = T


"""
Construct a PlaneWaveBasis from lattice vectors, a list of kpoints
and a list of coordinates in the reciprocal lattice

# Arguments
- `lattice`    Matrix-like object with one lattice vectors per column
- `kpoints`    Vector of Vector-like objects with dimensionality 3,
               which provides the list of k-points.
- `kweights`   Weights for the kpoints during a Brillouin-zone integration
- `Gcoords`    Vector of Vector-like objects with dimensionality 3,
               which provides the list of coordinates in reciprocal
               space to build the wave vectors of the plane waves
               in the basis.
- `Ecut`       Energy cutoff. Used to select the subset of wave vectors
               to be used at each k-Point, i.e. the basis set X_k.
"""
function PlaneWaveBasis(lattice::AbstractMatrix{T}, kpoints, kweights,
                        Gcoords::Vector{Vector{I}}, Ecut::Number
                       ) where {I <: Integer, T <: Real}
    @assert size(lattice) == (3, 3)
    @assert length(kpoints) == length(kweights)

    # Index of the DC component inside Gs
    idx_DC = findfirst(isequal([0, 0, 0]), Gcoords)
    @assert idx_DC != nothing

    # Build wavevectors
    lattice = Matrix{T}(lattice)
    recip_lattice = 2π * inv(lattice')
    Gs = [recip_lattice * coord for coord in Gcoords]

    # For each k-Point select the Gs to form the basis X_k of wave vectors
    # with energy below Ecut
    kmask = Vector{Vector{Int}}(undef, length(kpoints))
    qsq = Vector{Vector{T}}(undef, length(kpoints))
    for (ik, k) in enumerate(kpoints)
        # TODO Some duplicate work happens here
        kmask[ik] = findall(G -> sum(abs2, k + G) ≤ 2 * Ecut, Gs)
        qsq[ik] = map(G -> sum(abs2, G + k), Gs[kmask[ik]])
    end

    # Minimal and maximal coordinates along each direction
    min_coords = minimum(hcat(Gcoords...), dims=2)
    max_coords = maximum(hcat(Gcoords...), dims=2)

    # Form and optimise FFT grid dimensions
    fft_size = reshape(max_coords .- min_coords .+ 1, :)
    fft_size = optimise_fft_grid(fft_size)

    # Translation table from plane-wave to FFT grid
    idx_to_fft = [1 .+ mod.(coord, fft_size) for coord in Gcoords]

    # Form real-space grid Y*
    rmax = [ceil(Int, fft_size[idim] / 2) for idim in 1:3]
    rcmax = ceil.(Int, fft_size ./ 2)
    grid_Yst = Array{Vector{T}, 3}(undef, fft_size...)
    for klm in CartesianIndices(grid_Yst)
        rcoord = mod.([klm.I...] .- rcmax .- 1, fft_size) .- rcmax
        grid_Yst[klm] = lattice * (rcoord ./ fft_size)
    end

    # Normalisation of kweights:
    kweights = kweights / sum(kweights)

    # Plan a FFT, spending some time on finding an optimal algorithm
    # for the machine on which the computation runs
    tmp = Array{Complex{T}}(undef, fft_size...)
    fft_plan = plan_fft!(tmp, flags=FFTW.MEASURE)
    ifft_plan = plan_ifft!(tmp, flags=FFTW.MEASURE)

    @assert T <: Real
    PlaneWaveBasis{T}(lattice, recip_lattice, det(lattice),
                      kpoints, kweights, Gs, Ecut, idx_DC, kmask, qsq,
                      grid_Yst, idx_to_fft, fft_plan, ifft_plan)
end


"""
Construct a PlaneWaveBasis from lattice vectors and a list of kpoints

# Arguments
- `lattice`    Matrix-like object with one lattice vectors per column.
- `kpoints`    Vector of Vector-like objects with dimensionality 3,
               which provides the list of k-points.
- `kweights`   Weights for the kpoints during a Brillouin-zone integration
- `Ecut`       Energy cutoff. Used to select the subset of wave vectors
               to be used at each k-Point, i.e. the basis set X_k.
- `supersampling_Y`
  Supersampling factor between the plane-wave basis used for representing
  the orbitals and the basis for discretising the local potential and
  performing integrals between density and potential. For a numerically
  exact result the latter potential/density basis needs to be exactly
  twice as large, which is the default.
"""
function PlaneWaveBasis(lattice::AbstractMatrix{T}, kpoints, kweights, Ecut::Real;
                        supersampling_Y=2) where T <: Real
    # Build reciprocal lattice
    recip_lattice = 2π * inv(Matrix(lattice'))

    # We want the set of wavevectors {G} in the basis set X_k to be chosen such
    # that |G + k|^2/2 ≤ Ecut, i.e. |G + k| ≤ 2 * sqrt(Ecut). Additionally the
    # representation of the electron density and performing computation of
    # convolutions with local potentials requires more wavevectors to be
    # numerically exact. For this larger Y basis a supersampling factor of
    # supersampling_Y is chosen, such that the cutoff for |G|^2 for the
    # Y basis can be computed as
    cutoff_Gsq = 2 * supersampling_Y^2 * Ecut
    Gcoords = construct_pw_grid(recip_lattice, cutoff_Gsq, kpoints=kpoints)
    PlaneWaveBasis(lattice, kpoints, kweights, Gcoords, Ecut)
end


"""
Take an existing Plane-wave basis and replace its kpoints without altering
the plane-wave vectors, i.e. without altering the Gs
"""
function substitute_kpoints!(pw::PlaneWaveBasis{T}, kpoints, kweights) where T
    @assert length(kpoints) == length(kweights)

    # Substitute kpoints and kweights
    resize!(pw.kpoints, length(kpoints))
    resize!(pw.kweights, length(kweights))
    pw.kpoints[:] = kpoints
    pw.kweights[:] = kweights

    # Compute new kmask and substitute the old one
    resize!(pw.kmask, length(kpoints))
    resize!(pw.qsq, length(kpoints))
    for (ik, k) in enumerate(kpoints)
        # TODO Some duplicate work happens here
        pw.kmask[ik] = findall(G -> sum(abs2, k + G) ≤ 2 * pw.Ecut, pw.Gs)
        pw.qsq[ik] = map(G -> sum(abs2, G + k), pw.Gs[pw.kmask[ik]])
    end

    return pw
end


"""
Construct a plane-wave grid based on the passed reciprocal
lattice vectors recip_lattice is able to provide a discretisation
on all kpoints, such that at least the resulting cutoff is reached
at all points, that is that for all kpoints k and plane-wave vectors G,
we have |k + G|^2 ≤ cutoff_Gsq
"""
function construct_pw_grid(recip_lattice::Matrix, cutoff_Gsq::Number;
                           kpoints::Vector = [[0, 0, 0]])
    # For a particular k-Point, the coordinates [m n o] of the
    # complementory reciprocal lattice vectors satisfy
    #     |B * [m n o] + k|^2 ≤ cutoff_Gsq
    # Now
    #     |B * [m n o] + k| ≥ abs(|B * [m n o]| - |k|) = |B * [m n o]| - |k|
    # provided that |k| ≤ |B|, which is typically the case. Therefore
    #     |λ_{min}(B)| * |[m n o]| ≤ |B * [m n o]| ≤ sqrt(cutoff_Gsq) + |k|
    # (where λ_{min}(B) is the smallest eigenvalue of B), such that
    #     |[m n o]| ≤ (sqrt(cutoff_Gsq) + |k|) / |λ_{min}(B)|
    # In the extremal case, m = o = 0, such that
    #    n_max_trial = (sqrt(cutoff_Gsq) + |k|) / |λ_{min}(B)|

    eig_B = eigvals(recip_lattice)
    max_k = maximum(norm.(kpoints))

    # Check the assumption in above argument is true
    @assert max_k ≤ maximum(abs.(eig_B))

    # Use the above argument to figure out a trial upper bound n_max
    trial_n_max = ceil(Int, (max_k + sqrt(cutoff_Gsq)) / minimum(abs.(eig_B)))

    # Go over the trial range (extending trial_n_max by one for safety)
    trial_n_range = -trial_n_max-1:trial_n_max+1

    # Determine actual n_max
    n_max = 0
    for coord in CartesianIndices((trial_n_range, trial_n_range, trial_n_range))
        G = recip_lattice * [coord.I...]
        if any(sum(abs2, G + k) ≤ cutoff_Gsq for k in kpoints)
            @assert all(abs.([coord.I...]) .<= trial_n_max)
            n_max = max(n_max, maximum(abs.([coord.I...])))
        end
    end

    # Now fill returned quantities
    n_range = -n_max:n_max
    coords = [[coord.I...]
                 for coord in CartesianIndices((n_range, n_range, n_range)) ]
    coords = reshape(coords, :)
    return coords
end


"""
Optimise an FFT grid such that the number of grid points in each dimenion
agrees well with a fast FFT algorithm.
"""
function optimise_fft_grid(n_grid_points::Vector{Int})
    # Ensure a power of 2 in the number of grid points
    # in each dimension for a fast FFT
    # (because of the tree-like divide and conquer structure of the FFT)
    return map(x -> nextpow(2, x), n_grid_points)
end


#
# Perform FFT
#
"""
Perform an in-place FFT to translate between the X_k or X basis
and the real-space Y* grid.

# Arguments
- `pw`          PlaneWaveBasis object
- `f_fourier`   input argument, the representation of F on X or X_k
- `f_real`      output argument, the resulting representation of F on Y*
                (all data in this array will be destroyed)
- `idx_to_fft`  Map applied to translate f_fourier from the (smaller)
                orbital basis X or the k-point-specific part X_k to
                the (larger) potential basis Y by zero padding.
"""
function G_to_R!(pw::PlaneWaveBasis, f_fourier, f_real, idx_to_fft)
    @assert length(f_fourier) == length(idx_to_fft)
    @assert size(f_real) == size(pw.grid_Yst)

    # Pad the input data to reach the full size of the Y grid,
    # then perform an FFT Y -> Y*
    f_real .= 0
    for (ig, idx_fft) in enumerate(idx_to_fft)
        f_real[idx_fft...] = f_fourier[ig]
    end
    f_real = pw.iFFT * f_real  # Note: This destroys data in f_real

    # IFFT has a normalization factor of 1/length(ψ),
    # but the normalisation convention used in this code is
    # e_G(x) = e^iGx / sqrt(|Γ|), so we need to use the factor
    # below in order to match both conventions.
    f_real .*= prod(size(pw.grid_Yst))
end

"""
Perform an in-place FFT to translate between a representation of
a function f in the X basis and the real-space Y* grid.
All data in `f_Yst` will be destroyed.
"""
function X_to_Yst!(pw::PlaneWaveBasis, f_X, f_Yst)
    G_to_R!(pw, f_X, f_Yst, pw.idx_to_fft)
end

"""
Perform an in-place FFT to translate between a representation of
a function f in the X_k basis specific to the k-Point with index `ik`
and the real-space Y* grid.
All data in `f_Yst` will be destroyed.
"""
function Xk_to_Yst!(pw::PlaneWaveBasis, ik::Int, f_Xk, f_Yst)
    G_to_R!(pw, f_Xk, f_Yst, pw.idx_to_fft[pw.kmask[ik]])
end


"""
Perform an in-place iFFT to translate between the Y* grid and the
X_k or X basis.

# Arguments
- `pw`          PlaneWaveBasis object
- `f_real`      input argument, the representation of F on Y*
                (all data in this array will be destroyed)
- `f_fourier`   output argument, the resulting representation of F
                on X or X_k (all data in this array will be destroyed)
- `idx_to_fft`  Map applied to translate between the representation
                of f in the potential basis Y, i.e. the result from
                applying the discrete Fourier transform to `f_fourier`,
                and the smaller basis X_k or X by truncation.
"""
function R_to_G!(pw::PlaneWaveBasis, f_real, f_fourier, idx_to_fft)
    @assert length(f_fourier) == length(idx_to_fft)
    @assert size(f_real) == size(pw.grid_Yst)

    # Do FFT on the full FFT plan, but truncate the resulting frequency
    # range to the part defined by the idx_to_fft array
    f_fourier_extended = pw.FFT * f_real  # This destroys data in f_real
    f_fourier .= 0
    for (ig, idx_fft) in enumerate(idx_to_fft)
        f_fourier[ig] = f_fourier_extended[idx_fft...]
    end
    # Again adjust normalisation as in G_to_R
    f_fourier .*= 1 / prod(size(pw.grid_Yst))
end


"""
Perform an in-place iFFT to translate between a representation of
a function f on the Y* grid and the X basis
All data in `f_Yst` and `f_X` will be destroyed.
"""
function Yst_to_X!(pw::PlaneWaveBasis, f_Yst, f_X)
    R_to_G!(pw, f_Yst, f_X, pw.idx_to_fft)
end

"""
Perform an in-place iFFT to translate between a representation of
a function f in the Y* grid and the X_k basis specific to the
k-Point with index `ik`.
All data in `f_Yst` and `f_X` will be destroyed.
"""
function Yst_to_Xk!(pw::PlaneWaveBasis, ik::Int, f_Yst, f_Xk)
    R_to_G!(pw, f_Yst, f_Xk, pw.idx_to_fft[pw.kmask[ik]])
end
