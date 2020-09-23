include("fft.jl")

# There are two kinds of plane-wave basis sets used in DFTK.
# The k-dependent orbitals are discretized on spherical basis sets {G, 1/2 |k+G|^2 ≤ Ecut}.
# Potentials and densities are expressed on cubic basis sets large enough to contain
# products of orbitals. This also defines the real-space grid
# (as the dual of the cubic basis set).

"""
Discretization information for kpoint-dependent quantities such as orbitals.
More generally, a kpoint is a block of the Hamiltonian;
eg collinear spin is treated by doubling the number of kpoints.
"""
struct Kpoint{T <: Real}
    spin::Symbol                     # :up, :down, :both or :spinless
    coordinate::Vec3{T}              # Fractional coordinate of k-Point
    mapping::Vector{Int}             # Index of G_vectors[i] on the FFT grid:
                                     # G_vectors(basis)[kpt.mapping[i]] == G_vectors(kpt)[i]
    mapping_inv::Dict{Int, Int}      # Inverse of `mapping`:
                                     # G_vectors(basis)[i] = G_vectors(kpt)[kpt.mapping_inv[i]]
    G_vectors::Vector{Vec3{Int}}     # Wave vectors in integer coordinates:
                                     # ({G, 1/2 |k+G|^2 ≤ Ecut})
end


"""
The list of G vectors of a given `basis` or `kpoint`.
"""
G_vectors(kpt::Kpoint) = kpt.G_vectors


@doc raw"""
A plane-wave discretized `Model`.
Normalization conventions:
- Things that are expressed in the G basis are normalized so that if ``x`` is the vector,
  then the actual function is ``sum_G x_G e_G`` with
  ``e_G(x) = e^{iG x}/sqrt(unit_cell_volume)``.
  This is so that, eg ``norm(ψ) = 1`` gives the correct normalization.
  This also holds for the density and the potentials.
- Quantities expressed on the real-space grid are in actual values.

`G_to_r` and `r_to_G` convert between these representations.
"""
struct PlaneWaveBasis{T <: Real}
    model::Model{T}
    # the basis set is defined by {e_{G}, 1/2|k+G|^2 ≤ Ecut}
    Ecut::T

    # irreducible kpoints
    kpoints::Vector{Kpoint{T}}
    # BZ integration weights, summing up to 1
    # kweights[ik] = length(ksymops[ik]) / sum(length(ksymops[ik]) for ik=1:Nk)
    kweights::Vector{T}
    # ksymops[ikpt] is a list of symmetry operations (S,τ)
    # mapping to points in the reducible BZ
    ksymops::Vector{Vector{SymOp}}

    # fft_size defines both the G basis on which densities and
    # potentials are expanded, and the real-space grid
    fft_size::Tuple{Int, Int, Int}

    # Plans for forward and backward FFT
    opFFT  # out-of-place FFT plan
    ipFFT  # in-place FFT plan
    opIFFT
    ipIFFT

    # Instantiated terms (<: Term), that contain a backreference to basis.
    # See Hamiltonian for high-level usage
    terms::Vector{Any}

    # symmetry operations that leave the reducible Brillouin zone invariant.
    # Subset of model.symops, and superset of all the ksymops.
    # Independent of the `use_symmetry` option
    symops::Vector{SymOp}
end

# Default printing is just too verbose
Base.show(io::IO, basis::PlaneWaveBasis) =
    print(io, "PlaneWaveBasis (Ecut=$(basis.Ecut), $(length(basis.kpoints)) kpoints)")
Base.eltype(::PlaneWaveBasis{T}) where {T} = T

@timing function build_kpoints(model::Model{T}, fft_size, kcoords, Ecut; variational=true) where T
    model.spin_polarization in (:none, :collinear, :spinless) || (
        error("$(model.spin_polarization) not implemented"))
    spin = (:undefined,)
    if model.spin_polarization == :collinear
        spin = (:up, :down)
    elseif model.spin_polarization == :none
        spin = (:both, )
    elseif model.spin_polarization == :spinless
        spin = (:spinless, )
    end

    kpoints = Vector{Kpoint}()
    for k in kcoords
        k = Vec3{T}(k)  # rationals are sloooow
        mapping = Int[]
        Gvecs_k = Vec3{Int}[]
        # provide a rough hint so that the arrays don't have to be resized so much
        n_guess = div(prod(fft_size), 8)
        sizehint!(mapping, n_guess)
        sizehint!(Gvecs_k, n_guess)
        for (i, G) in enumerate(G_vectors(fft_size))
            if !variational || sum(abs2, model.recip_lattice * (G + k)) / 2 ≤ Ecut
                push!(mapping, i)
                push!(Gvecs_k, G)
            end
        end
        mapping_inv = Dict(ifull => iball for (iball, ifull) in enumerate(mapping))
        for σ in spin
            push!(kpoints, Kpoint(σ, k, mapping, mapping_inv, Gvecs_k))
        end
    end

    kpoints
end
build_kpoints(basis::PlaneWaveBasis, kcoords) =
    build_kpoints(basis.model, basis.fft_size, kcoords, basis.Ecut)

# This is the "internal" constructor; the higher-level one below should be preferred
@timing function PlaneWaveBasis(model::Model{T}, Ecut::Number,
                                kcoords::AbstractVector, ksymops, symops=nothing;
                                fft_size=nothing, variational=true,
                                optimize_fft_size=false, supersampling=2) where {T <: Real}
    if variational
        @assert Ecut > 0
        if fft_size === nothing
            fft_size = determine_fft_size(model, Ecut; supersampling=supersampling)
        end
    else
        # ensure fft_size is provided, and other options are not set
        # TODO make proper error messages when the interface gets a bit cleaned up
        @assert fft_size !== nothing
        @assert supersampling == 2
        @assert !optimize_fft_size
    end
    fft_size = Tuple{Int, Int, Int}(fft_size)

    if variational && optimize_fft_size
        # TODO this is a hack for now, we build the kpoints twice
        kpoints = build_kpoints(model, fft_size, kcoords, Ecut; variational=variational)
        fft_size = determine_fft_size_precise(model.lattice, Ecut, kpoints; supersampling=supersampling)
        fft_size = Tuple{Int, Int, Int}(fft_size)
    end

    # TODO generic FFT is kind of broken for some fft sizes
    #      ... temporary workaround, see more details in fft_generic.jl
    fft_size = next_working_fft_size.(T, fft_size)
    ipFFT, opFFT = build_fft_plans(T, fft_size)

    # The FFT interface specifies that fft has no normalization, and
    # ifft has a normalization factor of 1/length (so that both
    # operations are inverse to each other). The convention we want is
    # ψ(r) = sum_G c_G e^iGr / sqrt(Ω)
    # so that the ifft is normalized by 1/sqrt(Ω). It follows that the
    # fft must be normalized by sqrt(Ω) / length
    ipFFT *= sqrt(model.unit_cell_volume) / length(ipFFT)
    opFFT *= sqrt(model.unit_cell_volume) / length(opFFT)
    ipIFFT = inv(ipFFT)
    opIFFT = inv(opFFT)

    # Compute weights
    kweights = [length(symops) for symops in ksymops]
    kweights = T.(kweights) ./ sum(kweights)

    # Sanity checks
    @assert length(kcoords) == length(ksymops)
    max_E = sum(abs2, model.recip_lattice * floor.(Int, Vec3(fft_size) ./ 2)) / 2
    if variational && Ecut > max_E
        @warn("For a variational method, Ecut should be less than the maximal kinetic energy " *
              "the grid supports ($max_E)")
    end

    terms = Vector{Any}(undef, length(model.term_types))

    if symops === nothing
        # TODO instead compute the group generated by with ksymops, or
        # just retire this constructor. Not critical because this
        # should not be used in this context anyway...
        symops = vcat(ksymops...)
    end

    # Notice that this also builds index mapping from the k-point-specific basis
    # to the global basis and thus the fft_size needs to be final at this point.
    kpoints  = build_kpoints(model, fft_size, kcoords, Ecut; variational=variational)
    basis = PlaneWaveBasis{T}(
        model, Ecut, kpoints,
        kweights, ksymops, fft_size, opFFT, ipFFT, opIFFT, ipIFFT, terms, symops)

    # Instantiate terms
    for (it, t) in enumerate(model.term_types)
        term_name = string(nameof(typeof(t)))
        @timing "Instantiation $term_name" basis.terms[it] = t(basis)
    end
    basis
end

"""
Creates a new basis identical to `basis`, but with a different set of kpoints
"""
function PlaneWaveBasis(basis::PlaneWaveBasis, kcoords::AbstractVector,
                        ksymops::AbstractVector, symops=nothing)
    PlaneWaveBasis(basis.model, basis.Ecut, kcoords, ksymops, symops;
                   fft_size=basis.fft_size)
end


@doc raw"""
Creates a `PlaneWaveBasis` using the kinetic energy cutoff `Ecut` and a Monkhorst-Pack
kpoint grid. The MP grid can either be specified directly with `kgrid` providing the
number of points in each dimension and `kshift` the shift (0 or 1/2 in each direction).
If not specified a grid is generated using `kgrid_size_from_minimal_spacing` with
a minimal spacing of `2π * 0.022` per Bohr.

If `use_symmetry` is `true` (default) the symmetries of the
crystal are used to reduce the number of ``k``-Points which are
treated explicitly. In this case all guess densities and potential
functions must agree with the crystal symmetries or the result is
undefined.
"""
function PlaneWaveBasis(model::Model, Ecut::Number;
                        kgrid=kgrid_size_from_minimal_spacing(model.lattice, 2π * 0.022),
                        kshift=[iseven(nk) ? 1/2 : 0 for nk in kgrid],
                        use_symmetry=true,
                        kwargs...)
    if use_symmetry
        kcoords, ksymops, symops = bzmesh_ir_wedge(kgrid, model.symops, kshift=kshift)
    else
        kcoords, ksymops, _ = bzmesh_uniform(kgrid, kshift=kshift)
        # even when not using symmetry to reduce computations, still
        # store in symops the set of kgrid-preserving symops
        symops = symops_preserving_kgrid(model.symops, kcoords)
    end
    PlaneWaveBasis(model, Ecut, kcoords, ksymops, symops; kwargs...)
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
Return the list of r vectors, in reduced coordinates. By convention, this is in [0,1)^3.
"""
function r_vectors(basis::PlaneWaveBasis{T}) where T
    N1, N2, N3 = basis.fft_size
    (Vec3{T}(T(i-1) / N1, T(j-1) / N2, T(k-1) / N3) for i = 1:N1, j = 1:N2, k = 1:N3)
end

"""
Return the index tuple `I` such that `G_vectors(basis)[I] == G`
or the index `i` such that `G_vectors(kpoint)[i] == G`.
Returns nothing if outside the range of valid wave vectors.
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
        nothing  # Outside range of valid indices
    end
end

function index_G_vectors(basis::PlaneWaveBasis, kpoint::Kpoint,
                         G::AbstractVector{T}) where {T <: Integer}
    fft_size = basis.fft_size
    idx = index_G_vectors(basis, G)
    isnothing(idx) && return nothing
    idx_linear = LinearIndices(fft_size)[idx]
    get(kpoint.mapping_inv, idx_linear, nothing)
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
@timing_seq function G_to_r!(f_real::AbstractArray3, basis::PlaneWaveBasis,
                             f_fourier::AbstractArray3)
    mul!(f_real, basis.opIFFT, f_fourier)
end
@timing_seq function G_to_r!(f_real::AbstractArray3, basis::PlaneWaveBasis,
                             kpt::Kpoint, f_fourier::AbstractVector)
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
In-place version of `r_to_G!`.
NOTE: If `kpt` is given, not only `f_fourier` but also `f_real` is overwritten.
"""
@timing_seq function r_to_G!(f_fourier::AbstractArray3, basis::PlaneWaveBasis,
                             f_real::AbstractArray3)
    mul!(f_fourier, basis.opFFT, f_real)
end
@timing_seq function r_to_G!(f_fourier::AbstractVector, basis::PlaneWaveBasis,
                             kpt::Kpoint, f_real::AbstractArray3)
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
            ret[iG, ir] = cis(-2π * dot(r, G)) * sqrt(basis.model.unit_cell_volume) / prod(basis.fft_size)
        end
    end
    ret
end

""""
Convert a `basis` into one that uses or doesn't use BZ symmetrization
Mainly useful for debug purposes (e.g. in cases we don't want to
bother with symmetry)
"""
function PlaneWaveBasis(basis::PlaneWaveBasis; use_symmetry)
    use_symmetry && error("Not implemented")
    if all(s -> length(s) == 1, basis.ksymops)
        return basis
    end
    kcoords = []
    for (ik, kpt) in enumerate(basis.kpoints)
        for (S, τ) in basis.ksymops[ik]
            push!(kcoords, normalize_kpoint_coordinate(S * kpt.coordinate))
        end
    end
    new_basis = PlaneWaveBasis(basis.model, basis.Ecut, kcoords,
                               [[identity_symop()] for _ in 1:length(kcoords)];
                               fft_size=basis.fft_size)
end
