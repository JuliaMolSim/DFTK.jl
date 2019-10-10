# Contains the numerical specification of the model

using FFTW

struct Kpoint{T <: Real}
    spin::Symbol              # :up or :down
    coordinate::Vec3{T}       # Fractional coordinate of k-Point
    mapping::Vector{Int}      # Index of basis[i] on FFT grid
    basis::Vector{Vec3{Int}}  # Wave vectors in integer coordinates
end


struct PlaneWaveModel{T <: Real, TopFFT, TipFFT}
    model::Model{T}
    Ecut::T
    kpoints::Vector{Kpoint{T}}
    kweights::Vector{T}       # Brillouin zone integration weights
    ksymops::Vector{Vector{Tuple{Mat3{Int}, Vec3{T}}}}  # Symmetry operations per k-Point

    # Plans for forward and backward FFT on C_ρ
    fft_size::Tuple{Int, Int, Int}  # Using tuple here, since Vec3 splatting is slow
    opFFT::TopFFT  # out-of-place FFT
    ipFFT::TipFFT  # in-place FFT
end

"""
TODO docme
"""
function build_kpoints(basis::PlaneWaveModel{T}, kcoords; Ecut=basis.Ecut) where T
    model = basis.model

    model.spin_polarisation in [:none, :collinear] || (
        error("$(model.spin_polarisation) not implemented"))
    spin = [:undefined]
    if model.spin_polarisation == :collinear
        spin = [:up, :down]
    end

    kpoints = Vector{Kpoint{T}}()
    for k in kcoords
        energy(q) = sum(abs2, model.recip_lattice * q) / 2
        pairs = [(i, G) for (i, G) in enumerate(basis_Cρ(basis)) if energy(k + G) ≤ Ecut]

        for σ in spin
            push!(kpoints, Kpoint{T}(σ, k, first.(pairs), last.(pairs)))
        end
    end

    kpoints
end


"""
TODO docme

fft_size is now Fourier grid size
kcoords is vector of Vec3
"""
function PlaneWaveModel(model::Model{T}, fft_size, Ecut::Number,
                        kcoords, kweights, ksymops) where {T <: Real}
    fft_size = Tuple{Int, Int, Int}(fft_size)
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
    ipFFT = plan_fft!(tmp, flags=flags)
    opFFT = plan_fft(tmp, flags=flags)

    # IFFT has a normalization factor of 1/length(ψ),
    # but the normalisation convention used in this code is
    # e_G(x) = e^iGx / sqrt(|Γ|), so we scale the plans in-place
    # in order to match our convention.
    ipFFT *= 1 / length(ipFFT)
    opFFT *= 1 / length(opFFT)

    pw = PlaneWaveModel{T, typeof(opFFT), typeof(ipFFT)}(
        model, 0, [], [], [], fft_size, opFFT, ipFFT
    )
    PlaneWaveModel(pw, kcoords, kweights=kweights, ksymops=ksymops, Ecut=Ecut)
end

"""
TODO docme

Take an existing model and change the kpoints

For a consistent k-Point basis the kweights and ksymops should be updated accordingly.
If this is not done, density computation can give wrong results.

kcoords is vector of Vec3
"""
function PlaneWaveModel(pw::PlaneWaveModel{T, TopFFT, TipFFT}, kcoords;
                        kweights=nothing, ksymops=nothing,
                        Ecut=pw.Ecut) where {T, TopFFT, TipFFT}
    recip_lattice = pw.model.recip_lattice
    kweights === nothing && (kweights = ones(length(kcoords)) ./ length(kcoords))
    ksymops === nothing && (ksymops = [[(Mat3{Int}(I), Vec3(zeros(3)))]
                                         for _ in 1:length(kcoords)])
    @assert length(kcoords) == length(ksymops)
    @assert length(kcoords) == length(kweights)
    @assert sum(kweights) ≈ 1 "kweights are assumed to be normalized."
    max_E = sum(abs2, recip_lattice * floor.(Int, Vec3(pw.fft_size) ./ 2)) / 2
    @assert(Ecut ≤ max_E, "Ecut should be less than the maximal kinetic energy " *
            "the grid supports (== $max_E)")

    PlaneWaveModel{T, TopFFT, TipFFT}(pw.model, Ecut, build_kpoints(pw, kcoords; Ecut=Ecut),
                                      kweights, ksymops, pw.fft_size, pw.opFFT, pw.ipFFT)
end


"""
Return a generator producing the range of wave-vector coordinates contained
in the Fourier grid ``C_ρ`` described by the plane-wave basis in the correct order.
"""
function basis_Cρ(pw::PlaneWaveModel)
    start = -ceil.(Int, (Vec3(pw.fft_size) .- 1) ./ 2)
    stop  = floor.(Int, (Vec3(pw.fft_size) .- 1) ./ 2)
    axes = [[collect(0:stop[i]); collect(start[i]:-1)] for i in 1:3]
    (Vec3{Int}([i, j, k]) for i in axes[1], j in axes[2], k in axes[3])
end


"""
Return the index tuple corresponding to the wave vector in integer coordinates
in the ``C_ρ`` basis. Returns nothing if outside the range of valid wave vectors.
"""
function index_Cρ(pw::PlaneWaveModel, G::AbstractVector{T}) where {T <: Integer}
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
    G_to_r!(f_real, pw::PlaneWaveModel, [kpt::Kpoint, ], f_fourier)

Perform an iFFT to translate between `f_fourier`, a fourier representation
of a function either on ``B_k`` (if `kpt` is given) or on ``C_ρ`` (if not),
and `f_real`. The function will destroy all data in `f_real`.
"""
function G_to_r!(f_real::AbstractFFTGrid, pw::PlaneWaveModel, f_fourier::AbstractFFTGrid)
    n_bands = size(f_fourier, 4)
    for iband in 1:n_bands  # TODO Call batch version of FFTW, maybe do threading
        @views ldiv!(f_real[:, :, :, iband], pw.opFFT, f_fourier[:, :, :, iband])
    end
    real(f_real) .+ 0im  # TODO dirty hack for the moment
end
function G_to_r!(f_real::AbstractFFTGrid, pw::PlaneWaveModel, kpt::Kpoint,
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
    G_to_r(pw::PlaneWaveModel, [kpt::Kpoint, ], f_fourier)

Perform an iFFT to translate between `f_fourier`, a fourier representation
of a function either on ``B_k`` (if `kpt` is given) or on ``C_ρ`` (if not)
and return the values on the real-space grid `C_ρ^*`.
"""
function G_to_r(pw::PlaneWaveModel, f_fourier::AbstractFFTGrid)
    G_to_r!(similar(f_fourier), pw, f_fourier)
end
function G_to_r(pw::PlaneWaveModel, kpt::Kpoint, f_fourier::AbstractVector)
    G_to_r!(similar(f_fourier, pw.fft_size...), pw, kpt, f_fourier)
end
function G_to_r(pw::PlaneWaveModel, kpt::Kpoint, f_fourier::AbstractMatrix)
    G_to_r!(similar(f_fourier, pw.fft_size..., size(f_fourier, 2)), pw, kpt, f_fourier)
end



@doc raw"""
    r_to_G!(f_fourier, pw::PlaneWaveModel, [kpt::Kpoint, ], f_real)

Perform an FFT to translate between `f_real`, a function represented on
``C_ρ^\ast`` and its fourier representation. Truncatate the fourier
coefficients to ``B_k`` (if `kpt` is given). Note: If `kpt` is given, all data
in ``f_real`` will be distroyed as well.
"""
function r_to_G!(f_fourier::AbstractFFTGrid, pw::PlaneWaveModel, f_real::AbstractFFTGrid)
    n_bands = size(f_fourier, 4)
    for iband in 1:n_bands  # TODO Call batch version of FFTW, maybe do threading
        @views mul!(f_fourier[:, :, :, iband], pw.opFFT, f_real[:, :, :, iband])
    end
    f_fourier
end
function r_to_G!(f_fourier::AbstractVecOrMat, pw::PlaneWaveModel, kpt::Kpoint,
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
    r_to_G(pw::PlaneWaveModel, f_fourier)

Perform an FFT to translate between `f_fourier`, a fourier representation
on ``C_ρ^\ast`` and its fourier representation on ``C_ρ``.
"""
function r_to_G(pw::PlaneWaveModel, f_real::AbstractFFTGrid)
    r_to_G!(similar(f_real), pw, f_real)
end
# Note: There is deliberately no G_to_r version for the kpoints,
#       because at the moment this requires a copy of the input data f_real,
#       which is overwritten in r_to_G! for the k-Point version
