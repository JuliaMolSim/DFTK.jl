include("fft.jl")

# Contains the numerical specification of the model
#
# Normalization conventions:
# - Things that are expressed in the G basis are normalized so that if `x` is the vector,
#   then the actual function is `sum_G x_G e_G` with `e_G(x) = e^{iG x}/sqrt(unit_cell_volume)`.
#   This is so that, eg `norm(psi) = 1` gives the correct normalization.
#   This also holds for the density and the potentials.
# - Quantities expressed on the real-space grid are in actual values
#
# G_to_r and r_to_G convert between these.

# Each Kpoint has its own `basis`, consisting of all G vectors such that |k+G|^2 ≤ 1/2 Ecut
struct Kpoint{T <: Real}
    spin::Symbol              # :up, :down, :both or :spinless
    coordinate::Vec3{T}       # Fractional coordinate of k-Point
    mapping::Vector{Int}      # Index of basis[i] on FFT grid
    basis::Vector{Vec3{Int}}  # Wave vectors in integer coordinates
end

# fft_size defines both the G basis on which densities and potentials
# are expanded, and the real-space grid
#
# kpoints is the list of irreducible kpoints, kweights/ksymops contain
# the information needed to reconstruct the full BZ
struct PlaneWaveBasis{T <: Real, TopFFT, TipFFT}
    model::Model{T}
    Ecut::T
    kpoints::Vector{Kpoint{T}}
    kweights::Vector{T}       # Brillouin zone integration weights
    ksymops::Vector{Vector{Tuple{Mat3{Int}, Vec3{T}}}}  # Symmetry operations per k-Point

    # Plans for forward and backward FFT on C_ρ
    fft_size::Tuple{Int, Int, Int}  # Using tuple here, since Vec3 splatting is slow
                                    # (https://github.com/JuliaArrays/StaticArrays.jl/issues/361)
    opFFT::TopFFT  # out-of-place FFT plan
    ipFFT::TipFFT  # in-place FFT plan
end

"""
TODO docme
"""
function build_kpoints(model::Model{T}, fft_size, kcoords, Ecut) where T
    model.spin_polarisation in (:none, :collinear, :spinless) || (
        error("$(model.spin_polarisation) not implemented"))
    spin = (:undefined,)
    if model.spin_polarisation == :collinear
        spin = (:up, :down)
    elseif model.spin_polarisation == :none
        spin = (:both, )
    elseif model.spin_polarisation == :spinless
        spin = (:spinless, )
    end

    kpoints = Vector{Kpoint{T}}()
    for k in kcoords
        energy(q) = sum(abs2, model.recip_lattice * q) / 2
        pairs = [(i, G) for (i, G) in enumerate(basis_Cρ(fft_size)) if energy(k + G) ≤ Ecut]

        for σ in spin
            push!(kpoints, Kpoint{T}(σ, k, first.(pairs), last.(pairs)))
        end
    end

    kpoints
end
build_kpoints(basis::PlaneWaveBasis, kcoords) =
    build_kpoints(basis.model, basis.fft_size, kcoords, basis.Ecut)

"""
TODO docme

fft_size is now Fourier grid size
kcoords is vector of Vec3
"""
function PlaneWaveBasis(model::Model{T}, Ecut::Number,
                        kcoords::AbstractVector, ksymops=nothing, kweights=nothing;
                        fft_size=nothing) where {T <: Real}
    # TODO this constructor is too low-level. Write a hierarchy of
    # constructors starting at the high level
    # `PlaneWaveBasis(model, Ecut, kgrid)`

    @assert Ecut > 0
    if fft_size === nothing
        fft_size = determine_grid_size(model, Ecut)
    end
    fft_size = Tuple{Int, Int, Int}(fft_size)

    # TODO generic FFT is kind of broken for some fft sizes
    #      ... temporary workaround, see more details in fft.jl
    if !(T in [Float32, Float64]) && !all(is_fft_size_ok_for_generic.(fft_size))
        fft_size = next_working_fft_size_for_generic.(fft_size)
        @info "Changing fft size to $fft_size (smallest working size for generic FFTs)"
    end
    ipFFT, opFFT = build_fft_plans(T, fft_size)

    # The FFT interface specifies that fft has no normalization, and
    # ifft has a normalization factor of 1/length (so that both
    # operations are inverse to each other). The convention we want is
    # ψ(r) = sum_G c_G e^iGr / sqrt(Ω)
    # so that the ifft is normalized by 1/sqrt(Ω). It follows that the
    # fft must be normalized by sqrt(Ω)/length
    ipFFT *= sqrt(model.unit_cell_volume) / length(ipFFT)
    opFFT *= sqrt(model.unit_cell_volume) / length(opFFT)

    # Default to no symmetry
    ksymops === nothing && (ksymops = [[(Mat3{Int}(I), Vec3(zeros(3)))]
                                         for _ in 1:length(kcoords)])
    # Compute weights if not given
    if kweights === nothing
        kweights = [length(symops) for symops in ksymops]
        kweights = kweights / T(sum(kweights))
    end

    # Sanity checks
    @assert length(kcoords) == length(ksymops)
    @assert length(kcoords) == length(kweights)
    @assert sum(kweights) ≈ 1 "kweights are assumed to be normalized."
    max_E = sum(abs2, model.recip_lattice * floor.(Int, Vec3(fft_size) ./ 2)) / 2
    @assert(Ecut ≤ max_E, "Ecut should be less than the maximal kinetic energy " *
            "the grid supports (== $max_E)")

    PlaneWaveBasis{T, typeof(opFFT), typeof(ipFFT)}(
        model, Ecut, build_kpoints(model, fft_size, kcoords, Ecut),
        kweights, ksymops, fft_size, opFFT, ipFFT
    )
end

"""
Return a generator producing the range of wave-vector coordinates contained
in the Fourier grid ``C_ρ`` described by the plane-wave basis in the correct order.
"""
function basis_Cρ(fft_size)
    start = -ceil.(Int, (Vec3(fft_size) .- 1) ./ 2)
    stop  = floor.(Int, (Vec3(fft_size) .- 1) ./ 2)
    axes = [[collect(0:stop[i]); collect(start[i]:-1)] for i in 1:3]
    (Vec3{Int}([i, j, k]) for i in axes[1], j in axes[2], k in axes[3])
end
basis_Cρ(pw::PlaneWaveBasis) = basis_Cρ(pw.fft_size)


"""
Return the index tuple corresponding to the wave vector in integer coordinates
in the ``C_ρ`` basis. Returns nothing if outside the range of valid wave vectors.
"""
function index_Cρ(pw::PlaneWaveBasis, G::AbstractVector{T}) where {T <: Integer}
    start = -ceil.(Int, (Vec3(pw.fft_size) .- 1) ./ 2)
    stop  = floor.(Int, (Vec3(pw.fft_size) .- 1) ./ 2)
    lengths = stop .- start .+ 1

    function mapaxis(lengthi, Gi)
        Gi >= 0 && return 1 + Gi
        return 1 + lengthi + Gi
    end
    if all(start .<= G .<= stop)
        CartesianIndex(Tuple(mapaxis.(lengths, G)))
    else
        nothing # Outside range of valid indices
    end
end


#
# Perform FFT
#
# TODO Better name? => Put to types once decided
AbstractFFTGrid = Union{AbstractArray{T, 4}, AbstractArray{T, 3}} where T

@doc raw"""
    G_to_r!(f_real, pw::PlaneWaveBasis, [kpt::Kpoint, ], f_fourier)

Perform an iFFT to translate between `f_fourier`, a fourier representation
of a function either on ``B_k`` (if `kpt` is given) or on ``C_ρ`` (if not),
and `f_real`. The function will destroy all data in `f_real`.
"""
function G_to_r!(f_real::AbstractFFTGrid, pw::PlaneWaveBasis, f_fourier::AbstractFFTGrid)
    n_bands = size(f_fourier, 4)
    for iband in 1:n_bands  # TODO Call batch version of FFTW, maybe do threading
        @views ldiv!(f_real[:, :, :, iband], pw.opFFT, f_fourier[:, :, :, iband])
    end
    f_real
end
function G_to_r!(f_real::AbstractFFTGrid, pw::PlaneWaveBasis, kpt::Kpoint,
                 f_fourier::AbstractVecOrMat)
    n_bands = size(f_fourier, 2)
    @assert size(f_fourier, 1) == length(kpt.mapping)
    @assert size(f_real)[1:3] == pw.fft_size
    @assert size(f_real, 4) == n_bands

    f_real .= 0
    for iband in 1:n_bands  # TODO Call batch version of FFTW, maybe do threading
        # Pad the input data from B_k to C_ρ
        @views reshape(f_real, :, n_bands)[kpt.mapping, iband] = f_fourier[:, iband]

        # Perform an FFT on C_ρ -> C_ρ^*
        @views ldiv!(f_real[:, :, :, iband], pw.ipFFT, f_real[:, :, :, iband])
    end

    f_real
end

@doc raw"""
    G_to_r(pw::PlaneWaveBasis, [kpt::Kpoint, ], f_fourier)

Perform an iFFT to translate between `f_fourier`, a fourier representation
of a function either on ``B_k`` (if `kpt` is given) or on ``C_ρ`` (if not)
and return the values on the real-space grid `C_ρ^*`.
"""
function G_to_r(pw::PlaneWaveBasis, f_fourier::AbstractFFTGrid)
    G_to_r!(similar(f_fourier), pw, f_fourier)
end
function G_to_r(pw::PlaneWaveBasis, kpt::Kpoint, f_fourier::AbstractVector)
    G_to_r!(similar(f_fourier, pw.fft_size...), pw, kpt, f_fourier)
end
function G_to_r(pw::PlaneWaveBasis, kpt::Kpoint, f_fourier::AbstractMatrix)
    G_to_r!(similar(f_fourier, pw.fft_size..., size(f_fourier, 2)), pw, kpt, f_fourier)
end



@doc raw"""
    r_to_G!(f_fourier, pw::PlaneWaveBasis, [kpt::Kpoint, ], f_real)

Perform an FFT to translate between `f_real`, a function represented on
``C_ρ^\ast`` and its fourier representation. Truncatate the fourier
coefficients to ``B_k`` (if `kpt` is given). Note: If `kpt` is given, all data
in ``f_real`` will be distroyed as well.
"""
function r_to_G!(f_fourier::AbstractFFTGrid, pw::PlaneWaveBasis, f_real::AbstractFFTGrid)
    n_bands = size(f_fourier, 4)
    for iband in 1:n_bands  # TODO Call batch version of FFTW, maybe do threading
        @views mul!(f_fourier[:, :, :, iband], pw.opFFT, f_real[:, :, :, iband])
    end
    f_fourier
end
function r_to_G!(f_fourier::AbstractVecOrMat, pw::PlaneWaveBasis, kpt::Kpoint,
                 f_real::AbstractFFTGrid)
    n_bands = size(f_real, 4)
    @assert size(f_real)[1:3] == pw.fft_size
    @assert size(f_fourier, 1) == length(kpt.mapping)
    @assert size(f_fourier, 2) == n_bands

    f_fourier .= 0
    for iband in 1:n_bands  # TODO call batch version of FFTW, maybe do threading
        # FFT on C_ρ^∗ -> C_ρ
        @views mul!(f_real[:, :, :, iband], pw.ipFFT, f_real[:, :, :, iband])

        # Truncate the resulting frequency range to B_k
        @views f_fourier[:, iband] = reshape(f_real, :, n_bands)[kpt.mapping, iband]
    end
    f_fourier
end

@doc raw"""
    r_to_G(pw::PlaneWaveBasis, f_fourier)

Perform an FFT to translate between `f_fourier`, a fourier representation
on ``C_ρ^\ast`` and its fourier representation on ``C_ρ``.
"""
function r_to_G(pw::PlaneWaveBasis, f_real::AbstractFFTGrid)
    r_to_G!(similar(f_real), pw, f_real)
end
# Note: There is deliberately no G_to_r version for the kpoints,
#       because at the moment this requires a copy of the input data f_real,
#       which is overwritten in r_to_G! for the k-Point version
