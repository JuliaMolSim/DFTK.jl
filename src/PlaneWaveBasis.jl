include("fft.jl")

# There are two kinds of plane-wave basis sets used in DFTK.
# The k-dependent orbitals are discretized on spherical basis sets {G, 1/2 |k+G|^2 ≤ Ecut}
# Potentials and densities are expressed on cubic basis sets large
# enough to contain products of orbitals. This also defines the
# real-space grid (as the dual of the cubic basis set).

"""
Discretization information for kpoint-dependent quantities such as orbitals.
More generally, a kpoint is a block of the Hamiltonian;
eg collinear spin is treated by doubling the number of kpoints.
"""
struct Kpoint{T <: Real}
    spin::Symbol                  # :up, :down, :both or :spinless
    coordinate::Vec3{T}           # Fractional coordinate of k-Point
    mapping::Vector{Int}          # Index of G_vectors[i] on the FFT grid:
                                  # G_vectors(basis)[kpt.mapping[i]] == G_vectors(kpt)[i]
    G_vectors::Vector{Vec3{Int}}  # Wave vectors ({G, 1/2 |k+G|^2 ≤ Ecut}) in integer coordinates
end
"""
The list of G vectors of a given `basis` or `kpoint`.
"""
G_vectors(kpt::Kpoint) = kpt.G_vectors

"""
A plane-wave discretized `Model`.
Normalization conventions:
- Things that are expressed in the G basis are normalized so that if `x` is the vector,
  then the actual function is `sum_G x_G e_G` with `e_G(x) = e^{iG x}/sqrt(unit_cell_volume)`.
  This is so that, eg `norm(ψ) = 1` gives the correct normalization.
  This also holds for the density and the potentials.
- Quantities expressed on the real-space grid are in actual values.

G_to_r and r_to_G convert between these representations.
"""
struct PlaneWaveBasis{T <: Real, TopFFT, TipFFT, TopIFFT, TipIFFT}
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

    # Plans for forward and backward FFT
    opFFT::TopFFT  # out-of-place FFT plan
    ipFFT::TipFFT  # in-place FFT plan
    opIFFT::TopIFFT
    ipIFFT::TipIFFT

    # Instantiated terms (<: Term), that contain a backreference to basis.
    # See Hamiltonian for high-level usage
    terms::Vector{Any}
end
# Default printing is just too verbose
Base.show(io::IO, basis::PlaneWaveBasis) =
    print(io, "PlaneWaveBasis (Ecut=$(basis.Ecut), $(length(basis.kpoints)) kpoints)")
Base.eltype(basis::PlaneWaveBasis{T}) where {T} = T

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

function PlaneWaveBasis(model::Model{T}, Ecut::Number,
                        kcoords::AbstractVector, ksymops, kweights=nothing;
                        fft_size=determine_grid_size(model, Ecut)) where {T <: Real}
    @assert Ecut > 0
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
    ipIFFT = inv(ipFFT)
    opIFFT = inv(opFFT)

    # Compute default weights if not given
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

    terms = Vector{Any}(undef, length(model.term_types))

    basis = PlaneWaveBasis{T, typeof(opFFT), typeof(ipFFT), typeof(opIFFT), typeof(ipIFFT)}(
        model, Ecut, build_kpoints(model, fft_size, kcoords, Ecut),
        kweights, ksymops, fft_size, opFFT, ipFFT, opIFFT, ipIFFT, terms)

    # Instantiate terms
    for (it, t) in enumerate(model.term_types)
        basis.terms[it] = t(basis)
    end
    basis
end

"""
Creates a new basis identical to `basis`, but with a different set of kpoints
"""
function PlaneWaveBasis(basis::PlaneWaveBasis, kcoords::AbstractVector,
                        ksymops::AbstractVector, kweights=nothing)
    PlaneWaveBasis(basis.model, basis.Ecut, kcoords, ksymops, kweights;
                   fft_size=basis.fft_size)
end

function PlaneWaveBasis(model::Model, Ecut::Number;
                        kgrid=[1, 1, 1], enable_bzmesh_symmetry=true, kwargs...)
    if enable_bzmesh_symmetry
        kcoords, ksymops = bzmesh_ir_wedge(kgrid, model.lattice, model.atoms)
    else
        kcoords, ksymops = bzmesh_uniform(kgrid)
    end
    PlaneWaveBasis(model, Ecut, kcoords, ksymops; kwargs...)
end

"""
Return the list of wave vectors (integer coordinates) for the cubic basis set.
"""
function G_vectors(fft_size)
    start = -ceil.(Int, (Vec3(fft_size) .- 1) ./ 2)
    stop  = floor.(Int, (Vec3(fft_size) .- 1) ./ 2)
    axes = [[collect(0:stop[i]); collect(start[i]:-1)] for i in 1:3]
    (Vec3{Int}(i, j, k) for i in axes[1], j in axes[2], k in axes[3])
end
G_vectors(basis::PlaneWaveBasis) = G_vectors(basis.fft_size)

"""
Return the list of r vectors, in reduced coordinates. By convention, this is in [0,1]^3.
"""
function r_vectors(basis::PlaneWaveBasis{T}) where T
    N1, N2, N3 = basis.fft_size
    (Vec3{T}(T(i-1)/N1, T(j-1)/N2, T(k-1)/N3) for i = 1:N1, j = 1:N2, k = 1:N3)
end

"""
Return the index tuple I such that `G_vectors(basis)[I] == G`. Returns nothing if outside the range of valid wave vectors.
"""
function index_G_vectors(basis::PlaneWaveBasis, G::AbstractVector{T}) where {T <: Integer}
    start = -ceil.(Int, (Vec3(basis.fft_size) .- 1) ./ 2)
    stop  = floor.(Int, (Vec3(basis.fft_size) .- 1) ./ 2)
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
function G_to_r!(f_real::AbstractArray3, basis::PlaneWaveBasis,
                 f_fourier::AbstractArray3) where {Tr, Tf}
    mul!(f_real, basis.opIFFT, f_fourier)
end
function G_to_r!(f_real::AbstractArray3, basis::PlaneWaveBasis, kpt::Kpoint,
                 f_fourier::AbstractVector)
    @assert length(f_fourier) == length(kpt.mapping)
    @assert size(f_real) == basis.fft_size

    # Pad the input data
    fill!(f_real, 0)
    f_real[kpt.mapping] = f_fourier

    # Perform an FFT
    mul!(f_real, basis.ipIFFT, f_real)
end

"""
    G_to_r(basis::PlaneWaveBasis, [kpt::Kpoint, ] f_fourier)

Perform an iFFT to obtain the quantity defined by `f_fourier` defined
on the k-dependent spherical basis set (if `kpt` is given) or the
k-independent cubic (if it is not) on the real-space grid.
"""
function G_to_r(basis::PlaneWaveBasis, f_fourier::AbstractArray3)
    G_to_r!(similar(f_fourier), basis, f_fourier)
end
function G_to_r(basis::PlaneWaveBasis, kpt::Kpoint, f_fourier::AbstractVector)
    G_to_r!(similar(f_fourier, basis.fft_size...), basis, kpt, f_fourier)
end




@doc raw"""
In-place version of `r_to_G!`. NOTE: If `kpt` is given, not only `f_fourier` but also `f_real` is overwritten.
"""
function r_to_G!(f_fourier::AbstractArray3, basis::PlaneWaveBasis,
                 f_real::AbstractArray3)
    mul!(f_fourier, basis.opFFT, f_real)
end
function r_to_G!(f_fourier::AbstractVector, basis::PlaneWaveBasis, kpt::Kpoint,
                 f_real::AbstractArray3)
    @assert size(f_real) == basis.fft_size
    @assert length(f_fourier) == length(kpt.mapping)

    # FFT
    mul!(f_real, basis.ipFFT, f_real)

    # Truncate
    fill!(f_fourier, 0)
    f_fourier[:] = f_real[kpt.mapping]
end

"""
    r_to_G(basis::PlaneWaveBasis, [kpt::Kpoint, ] f_real)

Perform an FFT to obtain the Fourier representation of `f_real`. If
`kpt` is given, the coefficients are truncated to the k-dependent
spherical basis set.
"""
function r_to_G(basis::PlaneWaveBasis, f_real::AbstractArray3)
    r_to_G!(similar(f_real), basis, f_real)
end
# TODO optimize this
function r_to_G(basis::PlaneWaveBasis, kpt::Kpoint, f_real::AbstractArray3) where {T}
    r_to_G!(similar(f_real, length(kpt.mapping)), basis, kpt, copy(f_real))
end
