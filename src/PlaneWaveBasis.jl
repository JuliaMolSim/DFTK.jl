using MPI

# Abstract type for all possible bases that can be used in DFTK. Right now this is just
# one, but this type helps to resolve method ambiguities while avoiding an uninformative ::Any.
abstract type AbstractBasis{T <: Real} end

# There are two kinds of plane-wave basis sets used in DFTK.
# The k-dependent orbitals are discretized on spherical basis sets {G, 1/2 |k+G|^2 ≤ Ecut}.
# Potentials and densities are expressed on cubic basis sets large enough to contain
# products of orbitals. This also defines the real-space grid
# (as the dual of the cubic basis set).

"""
Discretization information for ``k``-point-dependent quantities such as orbitals.
More generally, a ``k``-point is a block of the Hamiltonian;
eg collinear spin is treated by doubling the number of kpoints.
"""
struct Kpoint{T <: Real}
    spin::Int                     # Spin component can be 1 or 2 as index into what is
                                  # returned by the `spin_components` function
    coordinate::Vec3{T}           # Fractional coordinate of k-point
    coordinate_cart::Vec3{T}      # Cartesian coordinate of k-point
    mapping::Vector{Int}          # Index of G_vectors[i] on the FFT grid:
                                  # G_vectors(basis)[kpt.mapping[i]] == G_vectors(basis, kpt)[i]
    mapping_inv::Dict{Int, Int}   # Inverse of `mapping`:
                                  # G_vectors(basis)[i] == G_vectors(basis, kpt)[mapping_inv[i]]
    G_vectors::Vector{Vec3{Int}}  # Wave vectors in integer coordinates:
                                  # ({G, 1/2 |k+G|^2 ≤ Ecut})
end

@doc raw"""
A plane-wave discretized `Model`.
Normalization conventions:
- Things that are expressed in the G basis are normalized so that if ``x`` is the vector,
  then the actual function is ``\sum_G x_G e_G`` with
  ``e_G(x) = e^{iG x} / \sqrt(\Omega)``, where ``\Omega`` is the unit cell volume.
  This is so that, eg ``norm(ψ) = 1`` gives the correct normalization.
  This also holds for the density and the potentials.
- Quantities expressed on the real-space grid are in actual values.

`G_to_r` and `r_to_G` convert between these representations.
"""
struct PlaneWaveBasis{T} <: AbstractBasis{T}
    model::Model{T}

    ## Global grid information
    # fft_size defines both the G basis on which densities and
    # potentials are expanded, and the real-space grid
    fft_size::Tuple{Int, Int, Int}
    # factor for integrals in real space: sum(ρ) * dvol ~ ∫ρ
    dvol::T  # = model.unit_cell_volume ./ prod(fft_size)
    # Information used to construct the k-point-specific basis
    # (not used directly after that)
    Ecut::T  # The basis set is defined by {e_{G}, 1/2|k+G|^2 ≤ Ecut}
    variational::Bool  # Is the k-point specific basis variationally consistent with
    #                    the basis used for the density / potential?

    ## Plans for forward and backward FFT
    # All these plans are *completely unnormalized* (eg FFT * BFFT != I)
    # The normalizations are performed in G_to_r/r_to_G according to
    # the DFTK conventions (see above)
    opFFT   # out-of-place FFT plan
    ipFFT   # in-place FFT plan
    opBFFT  # inverse plans (unnormalized plan; backward in FFTW terminology)
    ipBFFT
    r_to_G_normalization::T  # r_to_G = r_to_G_normalization * FFT
    G_to_r_normalization::T  # G_to_r = G_to_r_normalization * BFFT

    ## MPI-local information of the kpoints this processor treats
    # Irreducible kpoints. In the case of collinear spin,
    # this lists all the spin up, then all the spin down
    kpoints::Vector{Kpoint{T}}
    # BZ integration weights, summing up to model.n_spin_components
    kweights::Vector{T}
    # ksymops[ik] is a list of symmetry operations (S,τ)
    # mapping to points in the reducible BZ
    ksymops::Vector{Vector{SymOp}}

    ## MPI-global information of how the global k-point grid was constructed
    # Monkhorst-Pack grid used to generate the k-points, or nothing for custom k-points
    kgrid::Union{Nothing,Vec3{Int}}
    kshift::Union{Nothing,Vec3{T}}
    # full list of k-point coordinates; kpoints.coordinate is a subset of this
    # (possibly doubled because of spin)
    kcoords_global::Vector{Vec3{T}}
    ksymops_global::Vector{Vector{SymOp}}

    # Setup for MPI-distributed processing over k-points
    comm_kpts::MPI.Comm           # communicator for the kpoints distribution
    krange_thisproc::Vector{Int}  # indices of kpoints treated explicitly by this
    #                               processor in the global kcoords array
    krange_allprocs::Vector{Vector{Int}}  # indices of kpoints treated by the
    #                                       respective rank in comm_kpts

    # Symmetry operations that leave the reducible Brillouin zone invariant.
    # Subset of model.symmetries, and superset of all the ksymops.
    # Independent of the `use_symmetry` option
    symmetries::Vector{SymOp}

    # Instantiated terms (<: Term), that contain a backreference to basis.
    # See Hamiltonian for high-level usage
    terms::Vector{Any}
end


# prevent broadcast on pwbasis
import Base.Broadcast.broadcastable
Base.Broadcast.broadcastable(basis::PlaneWaveBasis) = Ref(basis)

Base.eltype(::PlaneWaveBasis{T}) where {T} = T

@timing function build_kpoints(model::Model{T}, fft_size, kcoords, Ecut;
                               variational=true) where T
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
                  Kpoint(iσ, k, model.recip_lattice * k, mapping, mapping_inv, Gvecs_k))
        end
    end

    vcat(kpoints_per_spin...)  # put all spin up first, then all spin down
end
function build_kpoints(basis::PlaneWaveBasis, kcoords)
    build_kpoints(basis.model, basis.fft_size, kcoords, basis.Ecut;
                  variational=basis.variational)
end

# Lowest-level constructor. All given parameters must be the same on all processors
# and are stored in PlaneWaveBasis for easy reconstruction
@timing function PlaneWaveBasis(model::Model{T},
                                Ecut::Number, fft_size, variational,
                                kcoords::AbstractVector, ksymops,
                                kgrid, kshift, symmetries, comm_kpts) where {T <: Real}
    if !(all(fft_size .== next_working_fft_size(T, fft_size)))
        error("Selected fft_size will not work for the buggy generic " *
              "FFT routines; use next_working_fft_size")
    end
    # need explicit convert in case it's given as array
    fft_size = Tuple{Int, Int, Int}(fft_size)
    MPI.Init()

    # Compute k-point information and spread them across processors
    # Right now we split only the kcoords: both spin channels have to be handled
    # by the same process
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
        # Note: MPI ranks are 0-based
        krange_allprocs = split_evenly(1:n_kpt, n_procs)
        krange_thisproc = krange_allprocs[1 + MPI.Comm_rank(comm_kpts)]
        @assert mpi_sum(length(krange_thisproc), comm_kpts) == n_kpt
        @assert !isempty(krange_thisproc)
    end
    kcoords_thisproc = kcoords[krange_thisproc]
    ksymops_thisproc = ksymops[krange_thisproc]

    # Setup fft_size and plans
    (ipFFT, opFFT, ipBFFT, opBFFT) = build_fft_plans(T, fft_size)

    # Normalization constants
    # r_to_G = r_to_G_normalization * FFT
    # The convention we want is
    # ψ(r) = sum_G c_G e^iGr / sqrt(Ω)
    # so that the G_to_r has to normalized by 1/sqrt(Ω).
    # The other constant is chosen because FFT * BFFT = N
    G_to_r_normalization = 1/sqrt(model.unit_cell_volume)
    r_to_G_normalization = sqrt(model.unit_cell_volume) / length(ipFFT)

    # Setup k-point basis sets
    !variational && @warn(
        "Non-variational calculations are experimental. " *
        "Not all features of DFTK may be supported or work as intended."
    )
    kpoints = build_kpoints(model, fft_size, kcoords_thisproc, Ecut; variational)
    # kpoints is now possibly twice the size of ksymops. Make things consistent
    if model.n_spin_components == 2
        ksymops_thisproc = vcat(ksymops_thisproc, ksymops_thisproc)
        krange_thisproc = vcat(krange_thisproc, n_kpt .+ krange_thisproc)
        krange_allprocs = [vcat(range, n_kpt .+ range) for range in krange_allprocs]
    end

    # Compute weights
    kweights = [length(symmetries) for symmetries in ksymops_thisproc]
    tot_weight = mpi_sum(sum(kweights), comm_kpts)
    kweights = T.(model.n_spin_components .* kweights) ./ tot_weight
    @assert mpi_sum(sum(kweights), comm_kpts) ≈ model.n_spin_components

    dvol  = model.unit_cell_volume ./ prod(fft_size)
    terms = Vector{Any}(undef, length(model.term_types))  # Dummy terms array, filled below
    basis = PlaneWaveBasis{T}(
        model, fft_size, dvol,
        Ecut, variational,
        opFFT, ipFFT, opBFFT, ipBFFT,
        r_to_G_normalization, G_to_r_normalization,
        kpoints, kweights, ksymops_thisproc, kgrid, kshift,
        kcoords, ksymops, comm_kpts, krange_thisproc, krange_allprocs,
        symmetries, terms)
    @assert length(kpoints) == length(kweights)

    # Instantiate the terms with the basis
    for (it, t) in enumerate(model.term_types)
        term_name = string(nameof(typeof(t)))
        @timing "Instantiation $term_name" basis.terms[it] = t(basis)
    end
    basis
end

# This is the "internal" constructor; the higher-level one below should be preferred
@timing function PlaneWaveBasis(model::Model{T}, Ecut::Number,
                                kcoords::AbstractVector, ksymops, symmetries=nothing;
                                fft_size=nothing, variational=true,
                                fft_size_algorithm=:fast, supersampling=2,
                                kgrid=nothing, kshift=nothing,
                                comm_kpts=MPI.COMM_WORLD) where {T <: Real}
    # Compute or validate fft_size
    if fft_size === nothing
        @assert variational
        fft_size = compute_fft_size(model::Model{T}, Ecut, kcoords;
                                    supersampling, algorithm=fft_size_algorithm)
    else
        # validate
        if variational
            max_E = sum(abs2, model.recip_lattice * floor.(Int, Vec3(fft_size) ./ 2)) / 2
            Ecut > max_E && @warn(
                "For a variational method, Ecut should be less than the maximal kinetic " *
                    "energy the grid supports ($max_E)"
            )
        else
            # ensure no other options are set
            @assert supersampling == 2
            @assert fft_size_algorithm == :fast
        end
    end

    # Validate kpoints and symmetries
    @assert length(kcoords) == length(ksymops)
    if symmetries === nothing
        # TODO instead compute the group generated by ksymops (also in
        # the other constructors). Not critical because this should
        # not be used in this context anyway...
        symmetries = vcat(ksymops...)
    end
    PlaneWaveBasis(model, Ecut, fft_size, variational, kcoords, ksymops,
                   kgrid, kshift, symmetries, comm_kpts)
end

"""
Creates a new basis identical to `basis`, but with a custom set of kpoints
"""
function PlaneWaveBasis(basis::PlaneWaveBasis, kcoords::AbstractVector,
                        ksymops::AbstractVector, symmetries=vcat(ksymops...))
    kgrid = kshift = nothing
    PlaneWaveBasis(basis.model, basis.Ecut,
                   basis.fft_size, basis.variational,
                   kcoords, ksymops, kgrid, kshift,
                   symmetries, basis.comm_kpts)
end


@doc raw"""
Creates a `PlaneWaveBasis` using the kinetic energy cutoff `Ecut` and a Monkhorst-Pack
``k``-point grid. The MP grid can either be specified directly with `kgrid` providing the
number of points in each dimension and `kshift` the shift (0 or 1/2 in each direction).
If not specified a grid is generated using `kgrid_from_minimal_spacing` with
a minimal spacing of `2π * 0.022` per Bohr.

If `use_symmetry` is `true` (default) the symmetries of the
crystal are used to reduce the number of ``k``-points which are
treated explicitly. In this case all guess densities and potential
functions must agree with the crystal symmetries or the result is
undefined.
"""
function PlaneWaveBasis(model::Model;
                        Ecut,
                        kgrid=kgrid_from_minimal_spacing(model, 2π * 0.022),
                        kshift=[iseven(nk) ? 1/2 : 0 for nk in kgrid],
                        use_symmetry=true,
                        kwargs...)
    if use_symmetry
        kcoords, ksymops, symmetries = bzmesh_ir_wedge(kgrid, model.symmetries; kshift)
    else
        kcoords, ksymops, _ = bzmesh_uniform(kgrid; kshift)
        # even when not using symmetry to reduce computations, still
        # store in symmetries the set of kgrid-preserving symmetries
        symmetries = symmetries_preserving_kgrid(model.symmetries, kcoords)
    end
    PlaneWaveBasis(model, austrip(Ecut), kcoords, ksymops, symmetries;
                   kgrid, kshift, kwargs...)
end


"""
    G_vectors(fft_size::Tuple)

The wave vectors `G` in reduced (integer) coordinates for a cubic basis set
of given sizes.
"""
function G_vectors(fft_size::Union{Tuple,AbstractVector})
    start = .- cld.(fft_size .- 1, 2)
    stop  = fld.(fft_size .- 1, 2)
    axes = [[collect(0:stop[i]); collect(start[i]:-1)] for i in 1:3]
    [Vec3{Int}(i, j, k) for i in axes[1], j in axes[2], k in axes[3]]
end

@doc raw"""
    G_vectors(basis::PlaneWaveBasis)
    G_vectors(basis::PlaneWaveBasis, kpt::Kpoint)

The list of wave vectors ``G`` in reduced (integer) coordinates of a `basis`
or a ``k``-point `kpt`.
"""
G_vectors(basis::PlaneWaveBasis) = G_vectors(basis.fft_size)
G_vectors(::PlaneWaveBasis, kpt::Kpoint) = kpt.G_vectors


@doc raw"""
    G_vectors_cart(basis::PlaneWaveBasis)
    G_vectors_cart(basis::PlaneWaveBasis, kpt::Kpoint)

The list of ``G`` vectors of a given `basis` or `kpt`, in cartesian coordinates.
"""
function G_vectors_cart(basis::PlaneWaveBasis)
    map(G -> basis.model.recip_lattice * G,
        G_vectors(basis.fft_size))
end
function G_vectors_cart(basis::PlaneWaveBasis, kpt::Kpoint)
    map(G -> basis.model.recip_lattice * G,
        G_vectors(basis, kpt))
end

@doc raw"""
    Gplusk_vectors(basis::PlaneWaveBasis, kpt::Kpoint)

The list of ``G + k`` vectors, in reduced coordinates.
"""
function Gplusk_vectors(basis::PlaneWaveBasis, kpt::Kpoint)
    map(G -> G+kpt.coordinate,
        G_vectors(basis, kpt))
end

@doc raw"""
    Gplusk_vectors_cart(basis::PlaneWaveBasis, kpt::Kpoint)

The list of ``G + k`` vectors, in cartesian coordinates.
"""
function Gplusk_vectors_cart(basis::PlaneWaveBasis, kpt::Kpoint)
    map(Gplusk -> basis.model.recip_lattice * Gplusk,
        Gplusk_vectors(basis, kpt))
end


@doc raw"""
    r_vectors(basis::PlaneWaveBasis)

The list of ``r`` vectors, in reduced coordinates. By convention, this is in [0,1)^3.
"""
function r_vectors(basis::PlaneWaveBasis{T}) where T
    N1, N2, N3 = basis.fft_size
    [Vec3{T}(T(i-1) / N1, T(j-1) / N2, T(k-1) / N3) for i = 1:N1, j = 1:N2, k = 1:N3]
end

@doc raw"""
    r_vectors_cart(basis::PlaneWaveBasis)

The list of ``r`` vectors, in cartesian coordinates.
"""
r_vectors_cart(basis::PlaneWaveBasis) = map(r -> basis.model.lattice * r,
                                            r_vectors(basis))


"""
Return the index tuple `I` such that `G_vectors(basis)[I] == G`
or the index `i` such that `G_vectors(basis, kpoint)[i] == G`.
Returns nothing if outside the range of valid wave vectors.
"""
function index_G_vectors(basis::PlaneWaveBasis, G::AbstractVector{T}) where {T <: Integer}
    start = .- cld.(basis.fft_size .- 1, 2)
    stop  = fld.(basis.fft_size .- 1, 2)
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
    res = sum(basis.kweights .* array)
    mpi_sum(res, basis.comm_kpts)
end


"""
Gather the distributed ``k``-point data on the master process and return
it as a `PlaneWaveBasis`. On the other (non-master) processes `nothing` is returned.
The returned object should not be used for computations and only to extract data
for post-processing and serialisation to disk.
"""
function gather_kpts(basis::PlaneWaveBasis)
    # No need to allocate and setup a new basis object
    mpi_nprocs(basis.comm_kpts) == 1 && return basis

    # Gather k-point info on master
    kcoords = getproperty.(basis.kpoints, :coordinate)
    kcoords = gather_kpts(kcoords, basis)
    ksymops = gather_kpts(basis.ksymops, basis)

    # Number of distinct k-point coordinates is number of k-points with spin 1
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
