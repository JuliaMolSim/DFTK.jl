using MPI
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
    model::Model{T}               # TODO Should be only lattice/atoms
    spin::Int                     # Spin component can be 1 or 2 as index into what is
                                  # returned by the `spin_components` function
    coordinate::Vec3{T}           # Fractional coordinate of k-Point
    coordinate_cart::Vec3{T}      # Cartesian coordinate of k-Point
    mapping::Vector{Int}          # Index of G_vectors[i] on the FFT grid:
                                  # G_vectors(basis)[kpt.mapping[i]] == G_vectors(kpt)[i]
    mapping_inv::Dict{Int, Int}   # Inverse of `mapping`:
                                  # G_vectors(basis)[i] = G_vectors(kpt)[kpt.mapping_inv[i]]
    G_vectors::Vector{Vec3{Int}}  # Wave vectors in integer coordinates:
                                  # ({G, 1/2 |k+G|^2 ≤ Ecut})
end


"""
The list of G vectors of a given `basis` or `kpoint`, in reduced coordinates.
"""
G_vectors(kpt::Kpoint) = kpt.G_vectors

"""
The list of G vectors of a given `basis` or `kpoint`, in cartesian coordinates.
"""
G_vectors_cart(kpt::Kpoint) = (kpt.model.recip_lattice * G for G in G_vectors(kpt))

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
    Ecut::T  # The basis set is defined by {e_{G}, 1/2|k+G|^2 ≤ Ecut}
    variational::Bool  # Is the k-Point specific basis variationally consistent with
    #                    the basis used for the density / potential?

    # Irreducible kpoints. In the case of collinear spin,
    # this lists all the spin up, then all the spin down
    kpoints::Vector{Kpoint{T}}
    # BZ integration weights, summing up to model.n_spin_components
    kweights::Vector{T}
    # ksymops[ik] is a list of symmetry operations (S,τ)
    # mapping to points in the reducible BZ
    ksymops::Vector{Vector{SymOp}}
    # Monkhorst-Pack grid used to generate the k-Points, or nothing for custom k-Points
    kgrid::Union{Nothing,Vec3{Int}}
    kshift::Union{Nothing,Vec3{T}}

    # Setup for MPI-distributed processing over k-Points
    comm_kpts::MPI.Comm           # communicator for the kpoints distribution
    krange_thisproc::Vector{Int}  # indices of kpoints treated explicitly by this
    #                               processor in the global kcoords array
    krange_allprocs::Vector{Vector{Int}}  # indices of kpoints treated by the
    #                                       respective rank in comm_kpts

    # fft_size defines both the G basis on which densities and
    # potentials are expanded, and the real-space grid
    fft_size::Tuple{Int, Int, Int}
    # factor for integrals in real space: sum(ρ) * dvol ~ ∫ρ
    dvol::T  # = model.unit_cell_volume ./ prod(fft_size)

    # Plans for forward and backward FFT
    # These plans follow DFTK conventions (see above)
    opFFT   # out-of-place FFT plan
    ipFFT   # in-place FFT plan
    opIFFT  # inverse plans
    ipIFFT

    # These are unnormalized plans (no normalization at all: BFFT*FFT != I)
    opFFT_unnormalized
    ipFFT_unnormalized
    opBFFT_unnormalized  # unnormalized IFFT, "backward" FFT in FFTW terminology
    ipBFFT_unnormalized

    # Instantiated terms (<: Term), that contain a backreference to basis.
    # See Hamiltonian for high-level usage
    terms::Vector{Any}

    # Symmetry operations that leave the reducible Brillouin zone invariant.
    # Subset of model.symmetries, and superset of all the ksymops.
    # Independent of the `use_symmetry` option
    symmetries::Vector{SymOp}
end

# prevent broadcast on pwbasis
import Base.Broadcast.broadcastable
Base.Broadcast.broadcastable(basis::PlaneWaveBasis) = Ref(basis)

# Default printing is just too verbose TODO This is too spartanic
Base.show(io::IO, basis::PlaneWaveBasis) =
    print(io, "PlaneWaveBasis (Ecut=$(basis.Ecut), $(length(basis.kpoints)) kpoints)")
Base.eltype(::PlaneWaveBasis{T}) where {T} = T

@timing function build_kpoints(model::Model{T}, fft_size, kcoords, Ecut; variational=true) where T
    kpoints_per_spin = [Kpoint[] for _ in 1:model.n_spin_components]
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
        for iσ = 1:model.n_spin_components
            push!(kpoints_per_spin[iσ],
                  Kpoint(model,  iσ, k, model.recip_lattice * k, mapping, mapping_inv, Gvecs_k))
        end
    end

    vcat(kpoints_per_spin...)  # put all spin up first, then all spin down
end
build_kpoints(basis::PlaneWaveBasis, kcoords) =
    build_kpoints(basis.model, basis.fft_size, kcoords, basis.Ecut)

# This is the "internal" constructor; the higher-level one below should be preferred
@timing function PlaneWaveBasis(model::Model{T}, Ecut::Number,
                                kcoords::AbstractVector, ksymops, symmetries=nothing;
                                fft_size=nothing, variational=true,
                                optimize_fft_size=false, supersampling=2,
                                kgrid=nothing, kshift=nothing,
                                comm_kpts=MPI.COMM_WORLD,
                               ) where {T <: Real}
    mpi_ensure_initialized()

    # Validate kpoints and symmetries
    @assert length(kcoords) == length(ksymops)
    if symmetries === nothing
        # TODO instead compute the group generated by with ksymops, or
        # just retire this constructor. Not critical because this
        # should not be used in this context anyway...
        symmetries = vcat(ksymops...)
    end

    # Compute kpoint information and spread them across processors
    # Right now we split only the kcoords: both spin channels have to be handled by the same process
    n_kpt   = length(kcoords)
    n_procs = mpi_nprocs(comm_kpts)
    if n_procs > n_kpt
        # XXX Supporting this would require fixing a bunch of "reducing over
        #     empty collections" errors
        if parse(Bool, get(ENV, "CI", "false"))
            # In the unit tests it is really annoying that this fails, but
            # generally it leads to duplicated work that is not in the users interest.
            comm_kpts = MPI.COMM_SELF
            krange_thisproc = 1:n_kpt
            krange_allprocs = fill(1:n_kpt, n_procs)
        else
            error("No point in trying to parallelize $n_kpt kpoints over $n_procs " *
                  "processes; reduce the number of MPI processes.")
        end
    else
        # get the slice of 1:n_kpt to be handled by this process
        krange_allprocs = split_evenly(1:n_kpt, n_procs)
        krange_thisproc = krange_allprocs[1 + MPI.Comm_rank(comm_kpts)]  # MPI ranks are 0-based
        @assert mpi_sum(length(krange_thisproc), comm_kpts) == n_kpt
        @assert !isempty(krange_thisproc)
    end
    kcoords = kcoords[krange_thisproc]
    ksymops = ksymops[krange_thisproc]

    # Setup fft_size and plans
    fft_size = validate_or_compute_fft_size(model::Model{T}, fft_size, Ecut, supersampling,
                                            variational, optimize_fft_size, kcoords)
    fft_size = mpi_max(fft_size, comm_kpts)
    (ipFFT_unnormalized,  opFFT_unnormalized,
     ipBFFT_unnormalized, opBFFT_unnormalized) = build_fft_plans(T, fft_size)

    # Normalize plans
    # The FFT interface specifies that fft has no normalization, and
    # ifft has a normalization factor of 1/length (so that both
    # operations are inverse to each other). The convention we want is
    # ψ(r) = sum_G c_G e^iGr / sqrt(Ω)
    # so that the ifft is normalized by 1/sqrt(Ω). It follows that the
    # fft must be normalized by sqrt(Ω) / length
    ipFFT = ipFFT_unnormalized * (sqrt(model.unit_cell_volume) / length(ipFFT_unnormalized))
    opFFT = opFFT_unnormalized * (sqrt(model.unit_cell_volume) / length(opFFT_unnormalized))
    ipIFFT = inv(ipFFT)
    opIFFT = inv(opFFT)

    # Setup kpoint basis sets
    !variational && @warn(
        "Non-variational calculations are experimental. " *
        "Not all features of DFTK may be supported or work as intended."
    )
    kpoints = build_kpoints(model, fft_size, kcoords, Ecut; variational=variational)
    # kpoints is now possibly twice the size of ksymops. Make things consistent
    if model.n_spin_components == 2
        ksymops = vcat(ksymops, ksymops)
        krange_thisproc = vcat(krange_thisproc, n_kpt .+ krange_thisproc)
        krange_allprocs = [vcat(range, n_kpt .+ range) for range in krange_allprocs]
    end

    # Compute weights
    kweights = [length(symmetries) for symmetries in ksymops]
    tot_weight = mpi_sum(sum(kweights), comm_kpts)
    kweights = T.(model.n_spin_components .* kweights) ./ tot_weight
    @assert mpi_sum(sum(kweights), comm_kpts) ≈ model.n_spin_components

    # Create dummy terms array for basis to handle
    terms = Vector{Any}(undef, length(model.term_types))

    dvol = model.unit_cell_volume ./ prod(fft_size)

    basis = PlaneWaveBasis{T}(
        model, Ecut, variational, kpoints,
        kweights, ksymops, kgrid, kshift, comm_kpts, krange_thisproc, krange_allprocs,
        fft_size, dvol, opFFT, ipFFT, opIFFT, ipIFFT,
        opFFT_unnormalized, ipFFT_unnormalized, opBFFT_unnormalized, ipBFFT_unnormalized,
        terms, symmetries)
    @assert length(kpoints) == length(kweights)

    # Instantiate the terms with the basis
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
                        ksymops::AbstractVector, symmetries=nothing)
    PlaneWaveBasis(basis.model, basis.Ecut, kcoords, ksymops, symmetries;
                   fft_size=basis.fft_size, variational=basis.variational)
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
function PlaneWaveBasis(model::Model, Ecut;
                        kgrid=kgrid_size_from_minimal_spacing(model.lattice, 2π * 0.022),
                        kshift=[iseven(nk) ? 1/2 : 0 for nk in kgrid],
                        use_symmetry=true, kwargs...)
    if use_symmetry
        kcoords, ksymops, symmetries = bzmesh_ir_wedge(kgrid, model.symmetries, kshift=kshift)
    else
        kcoords, ksymops, _ = bzmesh_uniform(kgrid, kshift=kshift)
        # even when not using symmetry to reduce computations, still
        # store in symmetries the set of kgrid-preserving symmetries
        symmetries = symmetries_preserving_kgrid(model.symmetries, kcoords)
    end
    PlaneWaveBasis(model, austrip(Ecut), kcoords, ksymops, symmetries;
                   kgrid=kgrid, kshift=kshift, kwargs...)
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
G_vectors_cart(basis::PlaneWaveBasis) = (basis.model.recip_lattice * G for G in G_vectors(basis.fft_size))

"""
Return the list of r vectors, in reduced coordinates. By convention, this is in [0,1)^3.
"""
function r_vectors(basis::PlaneWaveBasis{T}) where T
    N1, N2, N3 = basis.fft_size
    (Vec3{T}(T(i-1) / N1, T(j-1) / N2, T(k-1) / N3) for i = 1:N1, j = 1:N2, k = 1:N3)
end
"""
Return the list of r vectors, in cartesian coordinates.
"""
r_vectors_cart(basis::PlaneWaveBasis) = (basis.model.lattice * r for r in r_vectors(basis))

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

"""
Return the index range of ``k``-points that have a particular spin component.
"""
function krange_spin(basis::PlaneWaveBasis, spin::Integer)
    n_spin = basis.model.n_spin_components
    @assert 1 ≤ spin ≤ n_spin
    spinlength = div(length(basis.kpoints), n_spin)
    (1 + (spin - 1) * spinlength):(spin * spinlength)
end

"""
Sum an array over kpoints, taking weights into account
"""
function weighted_ksum(basis::PlaneWaveBasis, array)
    res = sum(@. basis.kweights * array)
    mpi_sum(res, basis.comm_kpts)
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
                             kpt::Kpoint, f_fourier::AbstractVector;
                             skip_normalization=false)
    plan = skip_normalization ? basis.ipBFFT_unnormalized : basis.ipIFFT
    @assert length(f_fourier) == length(kpt.mapping)
    @assert size(f_real) == basis.fft_size

    # Pad the input data
    fill!(f_real, 0)
    f_real[kpt.mapping] = f_fourier

    # Perform an FFT
    mul!(f_real, plan, f_real)
end

"""
    G_to_r(basis::PlaneWaveBasis, [kpt::Kpoint, ] f_fourier)

Perform an iFFT to obtain the quantity defined by `f_fourier` defined
on the k-dependent spherical basis set (if `kpt` is given) or the
k-independent cubic (if it is not) on the real-space grid.
"""
function G_to_r(basis::PlaneWaveBasis, f_fourier::AbstractArray; assume_real=true)
    # assume_real is true by default because this is the most common usage
    # (for densities & potentials)
    f_real = similar(f_fourier)
    @assert length(size(f_fourier)) ∈ (3, 4)
    # this exploits trailing index convention
    for iσ = 1:size(f_fourier, 4)
        @views G_to_r!(f_real[:, :, :, iσ], basis, f_fourier[:, :, :, iσ])
    end
    assume_real ? real(f_real) : f_real
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
    if isreal(f_real)
        f_real = complex.(f_real)
    end
    mul!(f_fourier, basis.opFFT, f_real)
end
@timing_seq function r_to_G!(f_fourier::AbstractVector, basis::PlaneWaveBasis,
                             kpt::Kpoint, f_real::AbstractArray3; skip_normalization=false)
    plan = skip_normalization ? basis.ipFFT_unnormalized : basis.ipFFT
    @assert size(f_real) == basis.fft_size
    @assert length(f_fourier) == length(kpt.mapping)

    # FFT
    mul!(f_real, plan, f_real)

    # Truncate
    f_fourier .= view(f_real, kpt.mapping)
end

"""
    r_to_G(basis::PlaneWaveBasis, [kpt::Kpoint, ] f_real)

Perform an FFT to obtain the Fourier representation of `f_real`. If
`kpt` is given, the coefficients are truncated to the k-dependent
spherical basis set.
"""
function r_to_G(basis::PlaneWaveBasis, f_real::AbstractArray)
    f_fourier = similar(f_real, complex(eltype(f_real)))
    @assert length(size(f_real)) ∈ (3, 4)
    # this exploits trailing index convention
    for iσ = 1:size(f_real, 4)
        @views r_to_G!(f_fourier[:, :, :, iσ], basis, f_real[:, :, :, iσ])
    end
    f_fourier
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


"""
Gather the distributed k-Point data on the master process and return
it as a `PlaneWaveBasis`. On the other (non-master) processes `nothing` is returned.
The returned object should not be used for computations and only to extract data
for post-processing and serialisation to disk.
"""
function gather_kpts(basis::PlaneWaveBasis)
    # No need to allocate and setup a new basis object
    mpi_nprocs(basis.comm_kpts) == 1 && return basis

    # Gather k-Point info on master
    kcoords = getproperty.(basis.kpoints, :coordinate)
    kcoords = gather_kpts(kcoords, basis)
    ksymops = gather_kpts(basis.ksymops, basis)

    # Number of distinct k-Point coordinates is number of k-Points with spin 1
    n_spinup_thisproc = count(kpt.spin == 1 for kpt in basis.kpoints)
    n_kcoords = mpi_sum(n_spinup_thisproc, basis.comm_kpts)

    if isnothing(kcoords)  # i.e. master process
        nothing
    else
        PlaneWaveBasis(basis.model,
                       basis.Ecut,
                       kcoords[1:n_kcoords],
                       ksymops[1:n_kcoords],
                       basis.symmetries;
                       fft_size=basis.fft_size,
                       kgrid=basis.kgrid,
                       kshift=basis.kshift,
                       variational=basis.variational,
                       comm_kpts=MPI.COMM_SELF,
                      )
    end
end


"""
Gather the distributed data of a quantity depending on `k`-Points on the master process
and return it. On the other (non-master) processes `nothing` is returned.
"""
function gather_kpts(data::AbstractArray, basis::PlaneWaveBasis)
    master = tag = 0
    n_kpts = sum(length, basis.krange_allprocs)

    if MPI.Comm_rank(basis.comm_kpts) == master
        allk_data = similar(data, n_kpts)
        allk_data[basis.krange_allprocs[1]] = data
        for rank in 1:mpi_nprocs(basis.comm_kpts) - 1  # Note: MPI ranks are 0-based
            rk_data, status = MPI.recv(rank, tag, basis.comm_kpts)
            @assert MPI.Get_error(status) == 0  # all went well
            allk_data[basis.krange_allprocs[rank + 1]] = rk_data
        end
        allk_data
    else
        MPI.send(data, master, tag, basis.comm_kpts)
        nothing
    end
end

# select the occupied orbitals assuming the Aufbau principle
function select_occupied_orbitals(basis::PlaneWaveBasis, ψ)
    model = basis.model
    n_spin = model.n_spin_components
    n_bands = div(model.n_electrons, n_spin * filled_occupation(model))
    [ψk[:, 1:n_bands] for ψk in ψ]
end

# Packing routines used in direct_minimization and newton algorithms.
# They pack / unpack sets of ψ's (or compatible arrays, such as hamiltonian
# applies and gradients) to make them compatible to be used in algorithms
# from IterativeSolvers.
# Some care is needed here : some operators (for instance K in newton.jl)
# are real-linear but not complex-linear. To overcome this difficulty, instead of
# seeing them as operators from C^N to C^N, we see them as
# operators from R^2N to R^2N. In practice, this is done with the
# reinterpret function from julia.
# /!\ pack_ψ does not share memory while unpack_ψ does

reinterpret_real(x) = reinterpret(real(eltype(x)), x)
reinterpret_complex(x) = reinterpret(Complex{eltype(x)}, x)

function pack_ψ(ψ)
    # TODO as an optimization, do that lazily? See LazyArrays
    vcat([vec(ψk) for ψk in ψ]...)
end

function unpack_ψ(x, sizes_ψ)
    n_bands = sizes_ψ[1][2]
    lengths = prod.(sizes_ψ)
    ends = cumsum(lengths)
    # We unsafe_wrap the resulting array to avoid a complicated type for ψ.
    # The resulting array is valid as long as the original x is still in live memory.
    map(1:length(sizes_ψ)) do ik
        unsafe_wrap(Array{complex(eltype(x))},
                    pointer(@views x[ends[ik]-lengths[ik]+1:ends[ik]]),
                    sizes_ψ[ik])
    end
end
