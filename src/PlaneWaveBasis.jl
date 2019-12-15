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

struct PlaneWaveBasis{T <: Real, Tgrid, TopFFT, TipFFT}
    model::Model{T}
    Ecut::T

    # kpoints is the list of irreducible kpoints
    kpoints::Vector{Kpoint{T}}
    # kweights/ksymops contain the information needed to reconstruct the full (reducible) BZ
    # Brillouin zone integration weights
    kweights::Vector{T}
    # ksymops[ikpt] is a list of symmetry operations (S,τ)
    ksymops::Vector{Vector{Tuple{Mat3{Int}, Vec3{T}}}}

    # fft_size defines both the G basis on which densities and
    # potentials are expanded, and the real-space grid
    fft_size::Tuple{Int, Int, Int}
    # real-space grids, in fractional coordinates.
    # grids[i] = 0, 1/Ni, ... (Ni-1)/Ni where Ni=fft_size[i]
    grids::Tgrid

    # Plans for forward and backward FFT on C_ρ
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
        pairs = [(i, G) for (i, G) in enumerate(G_vectors(fft_size)) if energy(k + G) ≤ Ecut]

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
        kweights = T.(kweights) ./ sum(kweights)
    end

    # Sanity checks
    @assert length(kcoords) == length(ksymops)
    @assert length(kcoords) == length(kweights)
    @assert sum(kweights) ≈ 1 "kweights are assumed to be normalized."
    max_E = sum(abs2, model.recip_lattice * floor.(Int, Vec3(fft_size) ./ 2)) / 2
    @assert(Ecut ≤ max_E, "Ecut should be less than the maximal kinetic energy " *
            "the grid supports (== $max_E)")

    grids = tuple((range(T(0), T(1), length=fft_size[i] + 1)[1:end-1] for i=1:3)...)
    PlaneWaveBasis{T, typeof(grids), typeof(opFFT), typeof(ipFFT)}(
        model, Ecut, build_kpoints(model, fft_size, kcoords, Ecut),
        kweights, ksymops, fft_size, grids, opFFT, ipFFT
    )
end

"""
Return a generator producing the range of wave-vector coordinates contained
in the Fourier grid described by the plane-wave basis in the correct order.
"""
function G_vectors(fft_size)
    start = -ceil.(Int, (Vec3(fft_size) .- 1) ./ 2)
    stop  = floor.(Int, (Vec3(fft_size) .- 1) ./ 2)
    axes = [[collect(0:stop[i]); collect(start[i]:-1)] for i in 1:3]
    (Vec3{Int}([i, j, k]) for i in axes[1], j in axes[2], k in axes[3])
end
G_vectors(pw::PlaneWaveBasis) = G_vectors(pw.fft_size)

"""
Return the index tuple I such that `G_vectors(pw)[I] == G`. Returns nothing if outside the range of valid wave vectors.
"""
function index_G_vectors(pw::PlaneWaveBasis, G::AbstractVector{T}) where {T <: Integer}
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
# Perform FFT.
#
# There are two kinds of FFT/IFFT functions: the ones on `C_ρ` (that
# don't take a kpoint as input) and the ones on `B_k` (that do)
# 
# Quantities in real-space are 3D arrays, quantities in reciprocal
# space are 3D (`C_ρ`) or 1D (`B_k`). We take as input either a single
# array (3D or 1D), or a bunch of them (4D or 2D)

AbstractArray3or4D = Union{AbstractArray{T, 3}, AbstractArray{T, 4}} where T
AbstractArray1or2D = Union{AbstractArray{T, 1}, AbstractArray{T, 2}} where T


# The methods below use the fact that in julia extra dimensions are ignored.
# Eg with x = randn(3), x[:,1] == x and size(x,2) == 1

@doc raw"""
    G_to_r!(f_real, pw::PlaneWaveBasis, [kpt::Kpoint, ], f_fourier)

Perform an iFFT to translate between `f_fourier`, a fourier representation
of a function either on ``B_k`` (if `kpt` is given) or on ``C_ρ`` (if not),
and `f_real`. The function will destroy all data in `f_real`.
"""
function G_to_r!(f_real::AbstractArray3or4D, pw::PlaneWaveBasis,
                 f_fourier::AbstractArray3or4D)
    n_bands = size(f_fourier, 4)
    @views Threads.@threads for iband in 1:n_bands
        ldiv!(f_real[:, :, :, iband], pw.opFFT, f_fourier[:, :, :, iband])
    end
    f_real
end
function G_to_r!(f_real::AbstractArray3or4D, pw::PlaneWaveBasis, kpt::Kpoint,
                 f_fourier::AbstractArray1or2D)
    n_bands = size(f_fourier, 2)
    @assert size(f_fourier, 1) == length(kpt.mapping)
    @assert size(f_real)[1:3] == pw.fft_size
    @assert size(f_real, 4) == n_bands


    @views Threads.@threads for iband in 1:n_bands
        fill!(f_real[:, :, :, iband], 0)
        # Pad the input data from B_k to C_ρ
        reshape(f_real, :, n_bands)[kpt.mapping, iband] = f_fourier[:, iband]

        # Perform an FFT on C_ρ -> C_ρ^*
        ldiv!(f_real[:, :, :, iband], pw.ipFFT, f_real[:, :, :, iband])
    end

    f_real
end

@doc raw"""
    G_to_r(pw::PlaneWaveBasis, [kpt::Kpoint, ], f_fourier)

Perform an iFFT to translate between `f_fourier`, a fourier representation
of a function either on ``B_k`` (if `kpt` is given) or on ``C_ρ`` (if not)
and return the values on the real-space grid `C_ρ^*`.
"""
function G_to_r(pw::PlaneWaveBasis, f_fourier::AbstractArray3or4D)
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
function r_to_G!(f_fourier::AbstractArray3or4D, pw::PlaneWaveBasis,
                 f_real::AbstractArray3or4D)
    n_bands = size(f_fourier, 4)
    @views Threads.@threads for iband in 1:n_bands
        mul!(f_fourier[:, :, :, iband], pw.opFFT, f_real[:, :, :, iband])
    end
    f_fourier
end
function r_to_G!(f_fourier::AbstractArray1or2D, pw::PlaneWaveBasis, kpt::Kpoint,
                 f_real::AbstractArray3or4D)
    n_bands = size(f_real, 4)
    @assert size(f_real)[1:3] == pw.fft_size
    @assert size(f_fourier, 1) == length(kpt.mapping)
    @assert size(f_fourier, 2) == n_bands

    @views Threads.@threads for iband in 1:n_bands
        # FFT on C_ρ^∗ -> C_ρ
        mul!(f_real[:, :, :, iband], pw.ipFFT, f_real[:, :, :, iband])

        # Truncate the resulting frequency range to B_k
        fill!(f_fourier[:, iband], 0)
        f_fourier[:, iband] = reshape(f_real, :, n_bands)[kpt.mapping, iband]
    end
    f_fourier
end

@doc raw"""
    r_to_G(pw::PlaneWaveBasis, f_fourier)

Perform an FFT to translate between `f_fourier`, a fourier representation
on ``C_ρ^\ast`` and its fourier representation on ``C_ρ``.
"""
function r_to_G(pw::PlaneWaveBasis, f_real::AbstractArray3or4D)
    r_to_G!(similar(f_real), pw, f_real)
end
# TODO optimize this
function r_to_G(pw::PlaneWaveBasis, kpt::Kpoint, f_real::AbstractArray{T, 3}) where {T}
    r_to_G!(similar(f_real, length(kpt.mapping)), pw, kpt, copy(f_real))
end
function r_to_G(pw::PlaneWaveBasis, kpt::Kpoint, f_real::AbstractArray{T, 4}) where {T}
    r_to_G!(similar(f_real, length(kpt.mapping), size(f_real, 4)), pw, kpt, copy(f_real))
end
