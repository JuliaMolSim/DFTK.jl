using FFTW

@doc raw"""
    PlaneWaveBasis(lattice::SMatrix{3, 3, T}, grid_size::SVector{3, I},
                   Ecut::Number, kpoints, kweights) where {T <: Real, I <: Integer}

Create a plane-wave basis.

## Examples
```julia-repl
julia> b = PlaneWaveBasis(TODO)
```

## Arguments and struct fields
- `lattice`: Real-space lattice vectors in columns
- `recip_lattice`: Reciprocal lattice vectors in columns
- `grid_size`:     Size of the rectangular Fourier grid used as the
                   density basis ``B_ρ``. In each dimension `idim` the
                   range of wave vectors (in integer coordinates) extends from
                   `-ceil(Int, (grid_size[idim]-1) / 2)` up to
                   `floor(Int, (grid_size[idim]-1) / 2)`. No optimisation is done
                   on the size of the grid with respect to performing FFTs.
- `Ecut`:          Kinetic energy cutoff in Hartree
- `kpoints`:       List of ``k``-Points in fractional coordinats
- `kweights`:      List of corresponding weights for the Brillouin-zone integration.
- `kbasis`:        List of wave vectors (in integer coordinates) for each ``k``-Point,
                   chosen such that ``|G + k|^2/2 \leq Ecut`` where ``G`` is the
                   wave vector. The defines the wave function basis ``B_{Ψ,k}`` for
                   each ``k``-Point.
- `FFT` and `iFFT`:   Containers for the forward and backward FFT plans on ``B_ρ``.
"""
struct PlaneWaveBasis{T <: Real}
    lattice::SMatrix{3, 3, T}
    recip_lattice::SMatrix{3, 3, T}
    unit_cell_volume::T
    recip_cell_volume::T

    grid_size::SVector{3, Int}
    Ecut::T
    kpoints::Vector{SVector{3, T}}
    kweights::Vector{T}
    kbasis::Vector{Vector{SVector{3, Int}}}

    FFT
    iFFT
end

function PlaneWaveBasis(lattice::AbstractMatrix{T}, grid_size,
                        Ecut::Number, kpoints, kweights) where {T <: Real}
    lattice = SMatrix{3, 3, T}(lattice)
    recip_lattice = 2π * inv(lattice')

    # Plan a FFT, spending some time on finding an optimal algorithm
    # for the machine on which the computation runs
    grid_size = SVector{3, Int}(grid_size)
    tmp = Array{Complex{T}}(undef, grid_size...)
    fft_plan = plan_fft!(tmp, flags=FFTW.MEASURE)
    ifft_plan = plan_ifft!(tmp, flags=FFTW.MEASURE)

    dummy_kpoints = Vector{SVector{3, T}}(undef, length(kpoints))
    dummy_kweights = Vector{T}(undef, length(kweights))
    dummy_kbasis = Vector{Vector{SVector{3, Int}}}(undef, length(kpoints))
    pw = PlaneWaveBasis{T}(lattice, recip_lattice, det(lattice), det(recip_lattice),
                           grid_size, Ecut, dummy_kpoints, dummy_kweights, dummy_kbasis,
                           fft_plan, ifft_plan)
    substitute_kpoints!(pw, kpoints, kweights)
end

"""
Replace the kpoints of an existing Plane-wave basis and make
the internal data structures consistent.
"""
function substitute_kpoints!(pw::PlaneWaveBasis{T}, kpoints, kweights) where T
    if length(kpoints) != length(kweights)
        error("Lengths of kpoints and length of kweights need to agree")
    end
    kweights = Vector{T}(kweights / sum(kweights))  # Normalise kweights
    kpoints = [SVector{3, T}(kp) for kp in kpoints]

    resize!(pw.kpoints, length(kpoints))[:] = kpoints
    resize!(pw.kweights, length(kweights))[:] = kweights

    # Update kbasis: For each k-Point select those Gcoords, such that
    # the kinetic energy cutoff is satisfied
    for (ik, k) in enumerate(kpoints)
        energy(Gc) = sum(abs2, pw.recip_lattice * (k + Gc)) / 2
        p = [Gc for Gc in gcoords(pw) if energy(Gc) ≤ pw.Ecut]
        pw.kbasis[ik] = [Gc for Gc in gcoords(pw) if energy(Gc) ≤ pw.Ecut]
    end

    pw
end

Base.eltype(basis::PlaneWaveBasis{T}) where T = T

"""
Return a generator producing the range of wave-vector coordinates contained
in the Fourier grid ``B_ρ`` described by the plane-wave basis.
"""
function gcoords(pw::PlaneWaveBasis)
    start = -ceil.(Int, (pw.grid_size .- 1) ./ 2)
    stop  = floor.(Int, (pw.grid_size .- 1) ./ 2)
    ci = CartesianIndices((UnitRange.(start, stop)..., ))
    (SVector(ci[i].I) for i in 1:length(ci))
end



#
# Perform FFT
#
# TODO Different function names for these, try to do a to-real transform
"""
Perform an in-place FFT to translate between the X_k or X basis
and the real-space Y* grid.

# Arguments
- `pw`          PlaneWaveBasis object
- `f_fourier`   input argument, the representation of F on X or X_k
- `f_real`      output argument, the resulting representation of F on Y*
                (all data in this array will be destroyed)
- `idx_to_fft`  Map applied to translate f_fourier from the
                density basis Y or the k-point-specific part of the
                (smaller) orbital basis X_k to the (larger) potential
                basis Y. In the case of X_k this involves zero padding
                as well.
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
a function f in the Y basis and the real-space Y* grid.
All data in `f_Yst` will be destroyed.
"""
function Y_to_Yst!(pw::PlaneWaveBasis, f_Y, f_Yst)
    G_to_R!(pw, f_Y, f_Yst, pw.idx_to_fft)
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
                and the basis X_k or Y. In the case of X_k, where the
                basis is smaller, this involves truncation as well.
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
a function f on the Y* grid and the Y basis
All data in `f_Yst` and `f_Y` will be destroyed.
"""
function Yst_to_Y!(pw::PlaneWaveBasis, f_Yst, f_Y)
    R_to_G!(pw, f_Yst, f_Y, pw.idx_to_fft)
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
