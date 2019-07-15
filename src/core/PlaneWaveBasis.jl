using FFTW

struct PlaneWaveBasis{T <: Real}
    # Lattice and reciprocal lattice vectors in columns
    lattice::Mat3{T}
    recip_lattice::Mat3{T}
    unit_cell_volume::T
    recip_cell_volume::T

    # Selected energy cutoff at construction time
    Ecut::T

    # Size of the rectangular Fourier grid used as the density basis
    # and the k-point-specific (spherical) wave function basis
    # with its wave vectors accessible per k-Point in the ordering
    # basis_wf[ik][iG]. Wave vectors are given in integer coordinates
    # and k-Points in fractional coordinates.
    grid_size::Vec3{Int}
    idx_DC::Int  # Index of the DC component in the rectangular grid
    kpoints::Vector{Vec3{T}}
    basis_wf::Vector{Vector{Vec3{Int}}}

    # Brillouin zone integration weights.
    kweights::Vector{T}

    # Plans for forward and backward FFT on B_ρ
    # TODO find a better way to type it; right now this is copypasted from
    # typeof(plan_fft!(zeros(ComplexF64,3,3,3)))
    # (resp plan_ifft)
    FFT::FFTW.cFFTWPlan{Complex{T},-1,true,3}
    iFFT::AbstractFFTs.ScaledPlan{Complex{Float64},FFTW.cFFTWPlan{Complex{T},1,true,3},T}
end

@doc raw"""
    PlaneWaveBasis(lattice::Mat3{T}, grid_size::Vec3{I}, Ecut::Number, kpoints,
                   kweights) where {T <: Real, I <: Integer}

Create a plane-wave basis from a specification for the Fourier grid size
and a kinetic energy cutoff to select the ``k``-point-specific wave function
basis ``B_{Ψ,k}`` in a way that the selected ``G`` wave vectors satisfy
``|G + k|^2/2 \leq Ecut``.

## Examples
```julia-repl
julia> b = PlaneWaveBasis(TODO)
```

## Arguments
- `lattice`:       Real-space lattice vectors in columns
- `grid_size`:     Size of the rectangular Fourier grid used as the
                   density basis ``B_ρ``. In each dimension `idim` the
                   range of wave vectors (in integer coordinates) extends from
                   `-ceil(Int, (grid_size[idim]-1) / 2)` up to
                   `floor(Int, (grid_size[idim]-1) / 2)`. No optimisation is done
                   on the size of the grid with respect to performing FFTs.
- `Ecut`:          Kinetic energy cutoff in Hartree
- `kpoints`:       List of ``k``-Points in fractional coordinats
- `kweights`:      List of corresponding weights for the Brillouin-zone integration.
"""
function PlaneWaveBasis(lattice::AbstractMatrix{T}, grid_size,
                        Ecut::Number, kpoints, kweights) where {T <: Real}
    lattice = SMatrix{3, 3, T, 9}(lattice)
    recip_lattice = 2π * inv(lattice')

    @assert(mod.(grid_size, 2) == ones(3),
            "grid_size needs to be a 3D Array with all entries odd, such that the " *
            "symmetry of a real quantity can be properly represented.")
    idx_DC = LinearIndices((Int.(grid_size)..., ))[ceil.(Int, grid_size ./ 2)...]

    # Plan a FFT, spending some time on finding an optimal algorithm
    # for the machine on which the computation runs
    fft_size = nextpow.(2, grid_size)  # Optimise FFT grid
    tmp = Array{Complex{T}}(undef, fft_size...)
    fft_plan = plan_fft!(tmp, flags=FFTW.MEASURE)
    ifft_plan = plan_ifft!(tmp, flags=FFTW.MEASURE)

    pw = PlaneWaveBasis{T}(lattice, recip_lattice, det(lattice), det(recip_lattice),
                           Ecut, grid_size, idx_DC, [], [], [], fft_plan, ifft_plan)
    set_kpoints!(pw, kpoints, kweights)
end

"""
Reset the kpoints of an existing Plane-wave basis and change the basis accordingly.
"""
function set_kpoints!(pw::PlaneWaveBasis{T}, kpoints, kweights; Ecut=pw.Ecut) where T
    @assert(length(kpoints) == length(kweights),
            "Lengths of kpoints and length of kweights need to agree")
    @assert sum(kweights) ≈ 1 "kweights are assumed to be normalized."
    max_E = sum(abs2, pw.recip_lattice * floor.(Int, pw.grid_size ./ 2)) / 2
    @assert(Ecut ≤ max_E, "Ecut should be less than the maximal kinetic energy " *
            "the grid supports (== $max_E)")

    resize!(pw.kpoints, length(kpoints)) .= kpoints
    resize!(pw.kweights, length(kweights)) .= kweights
    resize!(pw.basis_wf, length(kpoints))

    # Update basis_wf: For each k-Point select those G coords,
    # satisfying the energy cutoff
    for (ik, k) in enumerate(kpoints)
        energy(q) = sum(abs2, pw.recip_lattice * q) / 2
        p = [G for G in basis_ρ(pw) if energy(k + G) ≤ pw.Ecut]
        pw.basis_wf[ik] = [G for G in basis_ρ(pw) if energy(k + G) ≤ pw.Ecut]
    end
    pw
end

"""
Return a generator producing the range of wave-vector coordinates contained
in the Fourier grid ``B_ρ`` described by the plane-wave basis.
"""
function basis_ρ(pw::PlaneWaveBasis)
    start = -ceil.(Int, (pw.grid_size .- 1) ./ 2)
    stop  = floor.(Int, (pw.grid_size .- 1) ./ 2)
    (Vec3{Int}([i, j, k]) for i=start[1]:stop[1], j=start[2]:stop[2], k=start[3]:stop[3])
end


#
# Perform FFT
#
@doc raw"""
    G_to_r!(pw::PlaneWaveBasis, f_fourier, f_real[, basis_ρ])

Perform an in-place FFT to translate between `f_fourier`, a fourier representation
of a function using the wave vectors specified in `basis_ρ` and a representation
on the real-space density grid ``B^∗_ρ``. The function will destroy all data
in `f_real`. If `basis_ρ` is absent, the full density grid ``B_ρ`` is used.
"""
function G_to_r!(pw::PlaneWaveBasis, f_fourier, f_real; gcoords=basis_ρ(pw))
    @assert length(f_fourier) == length(gcoords)
    @assert(size(f_real) == size(pw.iFFT),
            "Size mismatch between f_real(==$(size(f_real)) and " *
            "FFT size(==$(size(pw.iFFT))")
    fft_size = [size(pw.FFT)...]

    # Pad the input data, then perform an FFT on the rectangular cube
    f_real .= 0
    for (ig, G) in enumerate(gcoords)
        idx_fft = 1 .+ mod.(G, fft_size)
        f_real[idx_fft...] = f_fourier[ig]
    end
    mul!(f_real, pw.iFFT, f_real)

    # IFFT has a normalization factor of 1/length(ψ),
    # but the normalisation convention used in this code is
    # e_G(x) = e^iGx / sqrt(|Γ|), so we need to use the factor
    # below in order to match both conventions.
    f_real .*= length(pw.iFFT)
end


@doc raw"""
    r_to_G!(pw::PlaneWaveBasis, f_real, f_fourier[, gcoords])

Perform an in-place FFT to translate between `f_real`, a representation of a
function on the real-space density grid ``B^∗_ρ`` and a fourier representation
using the wave vectors specified in `gcoords`. If `gcoords` is less than the
wave vectors required to exactly represent `f_real`, than this function implies
a truncation. On call all data in `f_real` and `f_fourier` will be destroyed.
If `gcoords` is absent, the full density grid ``B_ρ`` is used.
"""
function r_to_G!(pw::PlaneWaveBasis, f_real, f_fourier; gcoords=basis_ρ(pw))
    @assert length(f_fourier) == length(gcoords)
    @assert(size(f_real) == size(pw.FFT),
            "Size mismatch between f_real(==$(size(f_real)) and " *
            "FFT size(==$(size(pw.FFT))")
    fft_size = [size(pw.FFT)...]

    # Do FFT on the full FFT plan, but truncate the resulting frequency
    # range to the part defined by the idx_to_fft array
    f_fourier_extended = pw.FFT * f_real  # This destroys data in f_real
    f_fourier .= 0
    for (ig, G) in enumerate(gcoords)
        idx_fft = 1 .+ mod.(G, fft_size)
        f_fourier[ig] = f_fourier_extended[idx_fft...]
    end
    # Again adjust normalisation as in G_to_r
    f_fourier .*= 1 / length(pw.FFT)
end
