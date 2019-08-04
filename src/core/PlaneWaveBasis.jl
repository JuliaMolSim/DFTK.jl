using FFTW

struct PlaneWaveBasis{T <: Real, TFFT, TIFFT}
    # Lattice and reciprocal lattice vectors in columns
    lattice::Mat3{T}
    recip_lattice::Mat3{T}
    unit_cell_volume::T
    recip_cell_volume::T

    Ecut::T  # Selected energy cutoff at construction time

    # Size of the rectangular Fourier grid used as the density basis
    # and the k-point-specific (spherical) wave function basis
    # with its wave vectors accessible per k-Point in the ordering
    # basis_wf[ik][iG]. Wave vectors are given in integer coordinates
    # and k-Points in fractional coordinates.
    grid_size::Vec3{Int}
    idx_DC::Int  # Index of the DC component in the rectangular grid
    kpoints::Vector{Vec3{T}}
    basis_wf::Vector{Vector{Vec3{Int}}}
    kweights::Vector{T}  # Brillouin zone integration weights
    ksymops::Vector{Vector{Tuple{Mat3{Int}, Vec3{T}}}}  # Symmetry operations per k-Point


    # Plans for forward and backward FFT on B_ρ
    FFT::TFFT
    iFFT::TIFFT
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
- `kpoints`:       List of ``k``-Points in fractional coordinates
- `kweights`:      List of corresponding weights for the Brillouin-zone integration.
"""
function PlaneWaveBasis(lattice::AbstractMatrix{T}, grid_size,
                        Ecut::Number, kpoints, kweights, ksymops) where {T <: Real}
    lattice = SMatrix{3, 3, T, 9}(lattice)
    recip_lattice = 2π * inv(lattice')

    @assert(mod.(grid_size, 2) == ones(3),
            "grid_size needs to be a 3D Array with all entries odd, such that the " *
            "symmetry of a real quantity can be properly represented.")
    idx_DC = LinearIndices((Int.(grid_size)..., ))[ceil.(Int, grid_size ./ 2)...]

    # Optimise FFT grid: Make sure the obtained number factorises in small primes only
    fft_size = Vec3([nextprod([2, 3, 5], gs) for gs in grid_size])

    # Plan a FFT, spending some time on finding an optimal algorithm
    # for the machine on which the computation runs
    tmp = Array{Complex{T}}(undef, fft_size...)

    flags = FFTW.MEASURE
    if T == Float32
        flags |= FFTW.UNALIGNED
        # TODO For Float32 there are issues with aligned FFTW plans.
        #      Using unaligned FFTW plans is discouraged, but we do it anyways
        #      here as a quick fix. We should reconsider this in favour of using
        #      a parallel wisdom anyways in the future.
    end
    fft_plan = plan_fft!(tmp, flags=flags)
    ifft_plan = plan_ifft!(tmp, flags=flags)

    # IFFT has a normalization factor of 1/length(ψ),
    # but the normalisation convention used in this code is
    # e_G(x) = e^iGx / sqrt(|Γ|), so we scale the plans in-place
    # in order to match our convention.
    ifft_plan *= length(ifft_plan)
    fft_plan *= 1 / length(fft_plan)

    pw = PlaneWaveBasis{T, typeof(fft_plan), typeof(ifft_plan)}(
        lattice, recip_lattice, det(lattice), det(recip_lattice), Ecut, grid_size, idx_DC,
        [], [], [], [], fft_plan, ifft_plan
    )
    set_kpoints!(pw, kpoints, kweights=kweights, ksymops=ksymops)
end

"""
Reset the kpoints of an existing Plane-wave basis and change the basis accordingly.
For a consistent k-Point basis the kweights and ksymops should be updated accordingly.
If this is not done, density computation can give wrong results.
"""
function set_kpoints!(pw::PlaneWaveBasis{T}, kpoints; kweights=nothing, ksymops=nothing,
                      Ecut=pw.Ecut) where T
    kweights === nothing && (kweights = ones(length(kpoints)) ./ length(kpoints))
    ksymops === nothing && (ksymops = [[(Mat3{Int}(I), Vec3(zeros(3)))]
                                         for _ in 1:length(kpoints)])
    @assert(length(kpoints) == length(ksymops),
            "Lengths of kpoints and length of ksymops need to agree")
    @assert(length(kpoints) == length(kweights),
            "Lengths of kpoints and length of kweights need to agree")
    @assert sum(kweights) ≈ 1 "kweights are assumed to be normalized."
    max_E = sum(abs2, pw.recip_lattice * floor.(Int, pw.grid_size ./ 2)) / 2
    @assert(Ecut ≤ max_E, "Ecut should be less than the maximal kinetic energy " *
            "the grid supports (== $max_E)")

    resize!(pw.kpoints, length(kpoints)) .= kpoints
    resize!(pw.kweights, length(kweights)) .= kweights
    resize!(pw.ksymops, length(ksymops)) .= ksymops
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
function G_to_r!(pw::PlaneWaveBasis, f_fourier::AbstractVecOrMat, f_real::AbstractArray;
                 gcoords=basis_ρ(pw))
    fft_size = size(pw.FFT)
    n_bands = size(f_fourier, 2)
    @assert(size(f_fourier, 1) == length(gcoords), "Size mismatch between " *
            "f_fourier(==$(size(f_fourier))) and gcoords(==$(size(gcoords)))")
    @assert(size(f_real)[1:3] == fft_size,
            "Size mismatch between f_real(==$(size(f_real))) and " *
            "FFT size(==$(fft_size))")
    @assert(size(f_real, 4) == n_bands, "Differing number of bands in f_fourier and f_real")

    f_real .= 0
    for iband in 1:n_bands  # TODO Call batch version of FFTW, maybe do threading
        # Pad the input data
        for (ig, G) in enumerate(gcoords)
            idx_fft = 1 .+ mod.(G, Vec3(fft_size))
            # Tuple here because splatting SVectors directly is slow
            # (https://github.com/JuliaArrays/StaticArrays.jl/issues/361)
            f_real[Tuple(idx_fft)..., iband] = f_fourier[ig, iband]
        end

        # Perform an FFT on the rectangular cube
        # Note: normalization taken care of in the scaled plan
        @views mul!(f_real[:, :, :, iband], pw.iFFT, f_real[:, :, :, iband])
    end

    f_real
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
function r_to_G!(pw::PlaneWaveBasis, f_real::AbstractArray, f_fourier::AbstractVecOrMat;
                 gcoords=basis_ρ(pw))
    fft_size = size(pw.FFT)
    n_bands = size(f_fourier, 2)
    @assert(size(f_fourier, 1) == length(gcoords), "Size mismatch between " *
            "f_fourier(==$(size(f_fourier))) and gcoords(==$(size(gcoords)))")
    @assert(size(f_real)[1:3] == fft_size,
            "Size mismatch between f_real(==$(size(f_real))) and " *
            "FFT size(==$(fft_size))")
    @assert(size(f_real, 4) == n_bands, "Differing number of bands in f_fourier and f_real")

    f_fourier .= 0
    for iband in 1:n_bands  # TODO call batch version of FFTW, maybe do threading
        # Do FFT on rectangular cube
        # Note: normalization taken care of in the scaled plan
        @views mul!(f_real[:, :, :, iband], pw.FFT, f_real[:, :, :, iband])

        # Truncate the resulting frequency range to the part defined by the `gcoords`
        f_fourier_extended = f_real
        for (ig, G) in enumerate(gcoords)
            idx_fft = 1 .+ mod.(G, Vec3(fft_size))
            f_fourier[ig, iband] = f_fourier_extended[Tuple(idx_fft)..., iband]
        end
    end
    f_fourier
end
