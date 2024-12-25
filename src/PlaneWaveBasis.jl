import MPI

# Abstract type for all possible bases that can be used in DFTK. Right now this is just
# one, but this type helps to resolve method ambiguities while avoiding an uninformative ::Any.
abstract type AbstractBasis{T <: Real} end

# There are two kinds of plane-wave basis sets used in DFTK.
# The k-dependent orbitals are discretized on spherical basis sets {G, 1/2 |k+G|^2 ≤ Ecut}.
# Potentials and densities are expressed on cubic basis sets large enough to contain
# products of orbitals. This also defines the real-space grid
# (as the dual of the cubic basis set).

@doc raw"""
A plane-wave discretized `Model`.
Normalization conventions:
- Things that are expressed in the G basis are normalized so that if ``x`` is the vector,
  then the actual function is ``\sum_G x_G e_G`` with
  ``e_G(x) = e^{iG x} / \sqrt(\Omega)``, where ``\Omega`` is the unit cell volume.
  This is so that, eg ``norm(ψ) = 1`` gives the correct normalization.
  This also holds for the density and the potentials.
- Quantities expressed on the real-space grid are in actual values.

`ifft` and `fft` convert between these representations.
"""
struct PlaneWaveBasis{T,
                      VT <: Real,
                      Arch <: AbstractArchitecture,
                      FFTtype <: FFTGrid{T, VT},
                      T_kpt_G_vecs <: AbstractVector{Vec3{Int}},
                     } <: AbstractBasis{T}

    # T is the default type to express data, VT the corresponding bare value type (i.e. not dual)
    model::Model{T, VT}

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

    # A FFTGrid containing all necessary data for FFT opertations related to this basis
    fft_grid::FFTtype

    ## MPI-local information of the kpoints this processor treats
    # In principle, irreducible kpoints (although some kpoints might be duplicated in parallel runs).
    # In the case of collinear spin, this lists all the spin up, then all the spin down
    kpoints::Vector{Kpoint{T, T_kpt_G_vecs}}
    # BZ integration weights, summing up to model.n_spin_components
    kweights::Vector{T}

    ## (MPI-global) information on the k-point grid
    ## These fields are not actually used in computation, but can be used to reconstruct a basis
    # Monkhorst-Pack grid used to generate the k-points, or nothing for custom k-points
    kgrid::AbstractKgrid
    # Full list of (non spin doubled) k-point coordinates in the irreducible BZ (duplicates possible)
    # Best to use the irreducible_kcoords_global() and irreducible_kweights_global() functions 
    # to insure none of the k-points are duplicated
    kcoords_global::Vector{Vec3{T}}
    kweights_global::Vector{T}

    # Number of irreducible k-points in the basis. If there are more MPI ranks than irreducible
    # k-points, some are duplicated over the MPI ranks (with adjusted weight). In such a case
    # n_irreducible_kpoints < length(kcoords_global)
    n_irreducible_kpoints::Int

    ## Setup for MPI-distributed processing over k-points
    comm_kpts::MPI.Comm  # communicator for the kpoints distribution
    krange_thisproc::Vector{UnitRange{Int}}  # Indices of kpoints treated explicitly by this
    #            # processor in the global kcoords array. To allow for contiguous array
    #            # indexing, this is given as a unit range for spin-up and spin-down
    krange_allprocs::Vector{Vector{UnitRange{Int}}}  # Same as above, but one entry per rank
    krange_thisproc_allspin::Vector{Int}  # Indexing version == reduce(vcat, krange_thisproc)

    ## Information on the hardware and device used for computations.
    architecture::Arch

    ## Symmetry operations that leave the discretized model (k and r grids) invariant.
    # Subset of model.symmetries.
    symmetries::Vector{SymOp{VT}}
    # Whether the symmetry operations leave the rgrid invariant
    # If this is true, the symmetries are a property of the complete discretized model.
    # Therefore, all quantities should be symmetric to machine precision
    symmetries_respect_rgrid::Bool
    # Whether symmetry is used to reduce the number of explicit k-points to the
    # irreducible BZMesh. This is a debug option, useful when a part in the code does
    # not yet implement symmetry. See `unfold_bz` as a convenient way to use this.
    use_symmetries_for_kpoint_reduction::Bool

    ## Instantiated terms (<: Term). See Hamiltonian for high-level usage
    terms::Vector{Any}
end


# prevent broadcast
Base.Broadcast.broadcastable(basis::PlaneWaveBasis) = Ref(basis)

Base.eltype(::PlaneWaveBasis{T}) where {T} = T


function Kpoint(basis::PlaneWaveBasis, coordinate::AbstractVector, spin::Int)
    Kpoint(spin, coordinate, basis.model.recip_lattice, basis.fft_size, basis.Ecut;
           basis.variational, basis.architecture)
end


# Returns the kpoint at given coordinate. If outside the Brillouin zone, it is created
# from an equivalent kpoint in the basis (also returned)
function get_kpoint(basis::PlaneWaveBasis{T}, kcoord, spin) where {T}
    index, ΔG = find_equivalent_kpt(basis, kcoord, spin)
    equivalent_kpt = basis.kpoints[index]
    if iszero(ΔG)
        kpt = equivalent_kpt
    else
        kpt =  construct_from_equivalent_kpt(basis.fft_size, equivalent_kpt, kcoord, ΔG)
    end
    (; kpt, equivalent_kpt)
end

# Lowest-level constructor, should not be called directly.
# All given parameters must be the same on all processors
# and are stored in PlaneWaveBasis for easy reconstruction.
function PlaneWaveBasis(model::Model{T}, Ecut::Real, fft_size::Tuple{Int, Int, Int},
                        variational::Bool, kgrid::AbstractKgrid,
                        symmetries_respect_rgrid::Bool,
                        use_symmetries_for_kpoint_reduction::Bool,
                        comm_kpts, architecture::Arch
                       ) where {T <: Real, Arch <: AbstractArchitecture}
    # TODO This needs a refactor. There is too many different things here happening
    #      at once. In particular steps, which can become rather costly for larger
    #      calculations (symmetry determination, projector evaluation, potential
    #      evaluations) need to be redone ... even for cases (such as changing kpoints
    #      or going to a less accurate floating-point type) where parts of the
    #      computation could be avoided.
    #
    # Also we should allow for more flexibility regarding the floating-point
    # type and parallelisation, i.e. to temporarily change to a new FFT plan
    # or something like that.

    # Validate fft_size
    if variational
        max_E = norm2(model.recip_lattice * floor.(Int, Vec3(fft_size) ./ 2)) / 2
        Ecut > max_E && @warn(
            "For a variational method, Ecut should be less than the maximal kinetic " *
            "energy the grid supports ($max_E)"
        )
    end
    if !(all(fft_size .== next_working_fft_size(T, fft_size)))
        next_size = next_working_fft_size(T, fft_size)
        error("Selected fft_size=$fft_size will not work for the buggy generic " *
              "FFT routines; use next_working_fft_size(T, fft_size) = $next_size")
    end

    # Filter out the symmetries that don't preserve the real-space grid
    # and that don't preserve the k-point grid
    symmetries = model.symmetries
    if symmetries_respect_rgrid
        symmetries = symmetries_preserving_rgrid(symmetries, fft_size)
    end
    symmetries = symmetries_preserving_kgrid(symmetries, kgrid)

    # Build the irreducible k-point coordinates
    if use_symmetries_for_kpoint_reduction
        kdata = irreducible_kcoords(kgrid, symmetries)
    else
        kdata = irreducible_kcoords(kgrid, [one(SymOp)])
    end

    # Init MPI, and store MPI-global values for reference
    MPI.Init()
    kcoords_global  = convert(Vector{Vec3{T}}, kdata.kcoords)
    kweights_global = convert(Vector{T},       kdata.kweights)

    # Setup FFT plans
    fft_grid = FFTGrid(fft_size, model.unit_cell_volume, architecture) 

    # Compute k-point information and spread them across processors
    # Right now we split only the kcoords: both spin channels have to be handled
    # by the same process
    n_procs = mpi_nprocs(comm_kpts)
    n_kpt   = length(kcoords_global)
    n_irreducible_kpoints = n_kpt

    # The code cannot handle MPI ranks without k-points. If there are more prcocesses
    # than k-points, we duplicate k-points with the highest weight on the empty MPI
    # ranks (and scale the weight accordingly)
    if n_procs > n_kpt
        for i in n_kpt+1:n_procs
            idx = argmax(kweights_global)
            kweights_global[idx] *= 0.5
            push!(kweights_global, kweights_global[idx])
            push!(kcoords_global, kcoords_global[idx])
        end
        @warn("Attempting to parallelize $n_kpt k-points over $n_procs MPI ranks. " *
              "DFTK does not support processes empty of k-point. Some k-points were " *
              "duplicated over the extra ranks with scaled weights.")
    end
    n_kpt = length(kcoords_global)

    # get the slice of 1:n_kpt to be handled by this process
    # Note: MPI ranks are 0-based
    krange_allprocs1 = split_evenly(1:n_kpt, n_procs)
    krange_thisproc1 = krange_allprocs1[1 + MPI.Comm_rank(comm_kpts)]
    @assert mpi_sum(length(krange_thisproc1), comm_kpts) == n_kpt
    @assert !isempty(krange_thisproc1)

    # Setup k-point basis sets
    !variational && @warn(
        "Non-variational calculations are experimental. " *
        "Not all features of DFTK may be supported or work as intended."
    )
    kpoints = build_kpoints(model, fft_size, kcoords_global[krange_thisproc1], Ecut;
                            variational, architecture)
    # kpoints is now possibly twice the size of krange. Make things consistent
    if model.n_spin_components == 1
        kweights = kweights_global[krange_thisproc1]
        krange_allprocs = [[range] for range in krange_allprocs1]
    else
        kweights = vcat(kweights_global[krange_thisproc1],
                        kweights_global[krange_thisproc1])
        krange_allprocs = [[range, n_kpt .+ range] for range in krange_allprocs1]
    end
    krange_thisproc = krange_allprocs[1 + MPI.Comm_rank(comm_kpts)]
    krange_thisproc_allspin = reduce(vcat, krange_thisproc)

    @assert mpi_sum(sum(kweights), comm_kpts) ≈ model.n_spin_components
    @assert length(kpoints) == length(kweights)
    @assert length(kpoints) == sum(length, krange_thisproc)
    @assert length(kpoints) == length( krange_thisproc_allspin)

    if architecture isa GPU && Threads.nthreads() > 1
        error("Can't mix multi-threading and GPU computations yet.")
    end

    dvol  = model.unit_cell_volume ./ prod(fft_size)
    terms = Vector{Any}(undef, length(model.term_types))  # Dummy terms array, filled below

    basis = PlaneWaveBasis{T, value_type(T), Arch, typeof(fft_grid),
                           typeof(kpoints[1].G_vectors)}(
        model, fft_size, dvol,
        Ecut, variational,
        fft_grid,
        kpoints, kweights, kgrid,
        kcoords_global, kweights_global, n_irreducible_kpoints,
        comm_kpts, krange_thisproc, krange_allprocs, krange_thisproc_allspin,
        architecture, symmetries, symmetries_respect_rgrid,
        use_symmetries_for_kpoint_reduction, terms)

    # Instantiate the terms with the basis
    for (it, t) in enumerate(model.term_types)
        term_name = string(nameof(typeof(t)))
        @timing "Instantiation $term_name" basis.terms[it] = t(basis)
    end
    basis
end

@doc raw"""
Creates a `PlaneWaveBasis` using the kinetic energy cutoff `Ecut` for the Bloch waves
and a k-point grid.

By default a [`MonkhorstPack`](@ref) grid is employed, which can be specified as a
[`MonkhorstPack`](@ref) object or by simply passing a vector of three integers as
the `kgrid`. Optionally `kshift` allows to specify a shift (0 or 1/2 in each
direction). If not specified a grid is generated using `kgrid_from_maximal_spacing`
with a maximal spacing of `2π * 0.022` per Bohr.

By default the size of the FFT grid is automatically chosen from the kinetic energy
cutoff for the Bloch waves and a `supersampling` of 2.0 for the density. This is equal
to a density cutoff of ``\text{supersampling}^2 * \text{Ecut}`` or `4 * Ecut`.
A fully custom FFT grid size can also be chosen by specifying the `fft_size` parameter.
Note, this disables certain symmetry features.
"""
@timing function PlaneWaveBasis(model::Model{T};
                                Ecut::Number,
                                supersampling=2.0,
                                kgrid=nothing,
                                kshift=[0, 0, 0],
                                variational=true, fft_size=nothing,
                                symmetries_respect_rgrid=isnothing(fft_size),
                                use_symmetries_for_kpoint_reduction=true,
                                comm_kpts=MPI.COMM_WORLD, architecture=CPU()) where {T <: Real}
    if isnothing(fft_size)
        @assert variational
        # TODO Move this to compute_fft_size ?
        if symmetries_respect_rgrid
            # ensure that the FFT grid is compatible with the "reasonable" symmetries
            # (those with fractional translations with denominators 2, 3, 4, 6,
            #  this set being more or less arbitrary) by forcing the FFT size to be
            # a multiple of the denominators.
            # See https://github.com/JuliaMolSim/DFTK.jl/pull/642 for discussion
            denominators = [denominator(rationalize(sym.w[i]; tol=SYMMETRY_TOLERANCE))
                            for sym in model.symmetries for i = 1:3]
            factors = intersect((2, 3, 4, 6), denominators)
        else
            factors = (1, )
        end
        fft_size = compute_fft_size(model, Ecut, kgrid; supersampling, factors)
    else
        fft_size = Tuple{Int,Int,Int}(fft_size)
    end

    if isnothing(kgrid)
        kgrid_inner = kgrid_from_maximal_spacing(model, 2π * 0.022; kshift)
    elseif kgrid isa AbstractKgrid
        kgrid_inner = kgrid
    else
        kgrid_inner = MonkhorstPack(kgrid, kshift)
    end

    PlaneWaveBasis(model, austrip(Ecut), fft_size, variational, kgrid_inner,
                   symmetries_respect_rgrid, use_symmetries_for_kpoint_reduction,
                   comm_kpts, architecture)
end

"""
Creates a new basis identical to `basis`, but with a new k-point grid,
e.g. an [`MonkhorstPack`](@ref) or a [`ExplicitKpoints`](@ref) grid.
"""
@timing function PlaneWaveBasis(basis::PlaneWaveBasis, kgrid::AbstractKgrid)
    PlaneWaveBasis(basis.model, basis.Ecut,
                   basis.fft_size, basis.variational,
                   kgrid, basis.symmetries_respect_rgrid,
                   basis.use_symmetries_for_kpoint_reduction,
                   basis.comm_kpts, basis.architecture)
end


@doc raw"""
    G_vectors(basis::PlaneWaveBasis)
    G_vectors(basis::PlaneWaveBasis, kpt::Kpoint)

The list of wave vectors ``G`` in reduced (integer) coordinates of a `basis`
or a ``k``-point `kpt`.
"""
G_vectors(basis::PlaneWaveBasis) = basis.fft_grid.G_vectors
G_vectors(::PlaneWaveBasis, kpt::Kpoint) = kpt.G_vectors

@doc raw"""
    G_vectors_cart(basis::PlaneWaveBasis)
    G_vectors_cart(basis::PlaneWaveBasis, kpt::Kpoint)

The list of ``G`` vectors of a given `basis` or `kpt`, in Cartesian coordinates.
"""
function G_vectors_cart(basis::PlaneWaveBasis)
    map(recip_vector_red_to_cart(basis.model), G_vectors(basis))
end
function G_vectors_cart(basis::PlaneWaveBasis, kpt::Kpoint)
    recip_vector_red_to_cart.(basis.model, G_vectors(basis, kpt))
end

@doc raw"""
    Gplusk_vectors(basis::PlaneWaveBasis, kpt::Kpoint)

The list of ``G + k`` vectors, in reduced coordinates.
"""
function Gplusk_vectors(basis::PlaneWaveBasis, kpt::Kpoint)
    coordinate = kpt.coordinate  # Accelerator: avoid closure on kpt (not isbits)
    map(G -> G + coordinate, G_vectors(basis, kpt))
end

@doc raw"""
    Gplusk_vectors_cart(basis::PlaneWaveBasis, kpt::Kpoint)

The list of ``G + k`` vectors, in Cartesian coordinates.
"""
function Gplusk_vectors_cart(basis::PlaneWaveBasis, kpt::Kpoint)
    map(recip_vector_red_to_cart(basis.model), Gplusk_vectors(basis, kpt))
end

@doc raw"""
    r_vectors(basis::PlaneWaveBasis)

The list of ``r`` vectors, in reduced coordinates. By convention, this is in [0,1)^3.
"""
r_vectors(basis::PlaneWaveBasis) = basis.fft_grid.r_vectors

@doc raw"""
    r_vectors_cart(basis::PlaneWaveBasis)

The list of ``r`` vectors, in Cartesian coordinates.
"""
r_vectors_cart(basis::PlaneWaveBasis) = map(vector_red_to_cart(basis.model), r_vectors(basis))


"""
Return the index tuple `I` such that `G_vectors(basis)[I] == G`
or the index `i` such that `G_vectors(basis, kpoint)[i] == G`.
Returns nothing if outside the range of valid wave vectors.
"""
@inline function index_G_vectors(fft_size::Tuple, G::AbstractVector{<:Integer})
    # the inline declaration encourages the compiler to hoist these (G-independent) precomputations
    start = .- cld.(fft_size .- 1, 2)
    stop  = fld.(fft_size .- 1, 2)
    lengths = stop .- start .+ 1

    # FFTs store wavevectors as [0 1 2 3 -2 -1] (example for N=5)
    function G_to_index(length, G)
        G >= 0 && return 1 + G
        return 1 + length + G
    end
    if all(start .<= G .<= stop)
        CartesianIndex(Tuple(G_to_index.(lengths, G)))
    else
        nothing  # Outside range of valid indices
    end
end

# @inline is necessary here for the inner function to be inlined as well
@inline function index_G_vectors(basis::PlaneWaveBasis, G::AbstractVector{<:Integer})
    index_G_vectors(basis.fft_size, G)
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
    n_kpts_per_spin = div(length(basis.kpoints), n_spin)
    @assert 1 ≤ spin ≤ n_spin
    (1 + (spin - 1) * n_kpts_per_spin):(spin * n_kpts_per_spin)
end

"""
Sum an array over kpoints, taking weights into account
"""
function weighted_ksum(basis::PlaneWaveBasis, array)
    res = sum(basis.kweights .* array)
    mpi_sum(res, basis.comm_kpts)
end

"""
Utilities to get information about the irreducible k-point mesh (in case of duplication)
Useful for I/O, where k-point information should not be duplicated
"""
function irreducible_kcoords_global(basis::PlaneWaveBasis)
    # Assume that duplicated k-points are appended at the end of the kcoords array
    basis.kcoords_global[1:basis.n_irreducible_kpoints]
end

function irreducible_kweights_global(basis::PlaneWaveBasis{T}) where {T}
    function same_kpoint(i_irr, i_dupl)
        maximum(abs, basis.kcoords_global[i_dupl]-basis.kcoords_global[i_irr]) < eps(T)
    end

    # Check that weights add up to 1 on entry (non spin doubled k-points)
    @assert sum(basis.kweights_global) ≈ 1

    # Assume that duplicated k-points are appended at the end of the kcoords array
    irr_kweights = basis.kweights_global[1:basis.n_irreducible_kpoints]
    for i_dupl = basis.n_irreducible_kpoints+1:length(basis.kweights_global)
        for i_irr = 1:basis.n_irreducible_kpoints
            if same_kpoint(i_irr, i_dupl)
                irr_kweights[i_irr] += basis.kweights_global[i_dupl]
                break
            end
        end
    end

    # Test that irreducible weight add up to 1 (non spin doubled k-points)
    @assert sum(irr_kweights) ≈ 1
    irr_kweights
end

"""
Gather the distributed ``k``-point data on the master process and return
it as a `PlaneWaveBasis`. On the other (non-master) processes `nothing` is returned.
The returned object should not be used for computations and only for debugging
or to extract data for serialisation to disk.
"""
function gather_kpts(basis::PlaneWaveBasis)
    # No need to allocate and setup a new basis object
    mpi_nprocs(basis.comm_kpts) == 1 && return basis

    if mpi_master()
        PlaneWaveBasis(basis.model,
                       basis.Ecut,
                       basis.fft_size,
                       basis.variational,
                       basis.kgrid,
                       basis.symmetries_respect_rgrid,
                       basis.use_symmetries_for_kpoint_reduction,
                       MPI.COMM_SELF,
                       basis.architecture)
    else
        nothing
    end
end


"""
Gather the distributed data of a quantity depending on `k`-Points on the master process
and save it in `dest` as a dense `(size(kdata[1])..., n_kpoints)` array. On the other
(non-master) processes `nothing` is returned.
"""
@views function gather_kpts_block!(dest, basis::PlaneWaveBasis, kdata::AbstractVector{A}) where {A}
    # Number of elements stored per k-point in `kdata` (as vector of arrays)
    n_chunk = MPI.Bcast(length(kdata[1]), 0, basis.comm_kpts)
    @assert all(length(k) == n_chunk for k in kdata)

    # Note: This function assumes that k-points are stored contiguously in rank-increasing
    # order, i.e. it depends on the splitting realised by split_evenly.
    # Note that if some k-points are duplicated over MPI ranks, they are also gathered here.
    for σ in 1:basis.model.n_spin_components
        if mpi_master(basis.comm_kpts)
            # Setup variable buffer using appropriate data lengths and 
            counts = [n_chunk * length(basis.krange_allprocs[rank][σ])
                      for rank in 1:mpi_nprocs(basis.comm_kpts)]
            displs = [n_chunk * (first(basis.krange_allprocs[rank][σ])-1)
                      for rank in 1:mpi_nprocs(basis.comm_kpts)]
            @assert all(displs .+ counts .≤ length(dest))
            @assert eltype(dest) == eltype(A)
            destbuf = MPI.VBuffer(dest, counts, displs)
        else
            destbuf = nothing
        end

        # Make contiguous send buffer from vector of k-point-specific data
        sendbuf = kdata[krange_spin(basis, σ)]
        if ndims(A) > 0  # Scalar
            sendbuf = reduce((v, w) -> cat(v, w; dims=ndims(A) + 1), sendbuf)
        end
        MPI.Gatherv!(sendbuf, destbuf, basis.comm_kpts)
    end
    dest
end
function gather_kpts_block(basis::PlaneWaveBasis, kdata::AbstractVector{A}) where {A}
    dest = nothing
    if mpi_master(basis.comm_kpts)
        n_kptspin = length(basis.kcoords_global) * basis.model.n_spin_components
        dest = zeros(eltype(A), size(kdata[1])..., n_kptspin)
    end
    gather_kpts_block!(dest, basis, kdata)
end

"""
Scatter the data of a quantity depending on `k`-Points from the master process
to the child processes and return it as a Vector{Array}, where the outer vector
is a list over all k-points. On non-master processes `nothing` may be passed.
"""
function scatter_kpts_block(basis::PlaneWaveBasis, data::Union{Nothing,AbstractArray})
    T, N = (mpi_master(basis.comm_kpts) ? (eltype(data), ndims(data))
                                        : (nothing, nothing))
    T, N = MPI.bcast((T, N), 0, basis.comm_kpts)
    splitted = Vector{Array{T,N-1}}(undef, length(basis.kpoints))

    for σ in 1:basis.model.n_spin_components
        # Setup variable buffer for sending using appropriate data lengths
        if mpi_master(basis.comm_kpts)
            @assert data isa AbstractArray
            chunkshape = size(data)[1:end-1]
            n_chunk = prod(chunkshape; init=one(Int))
            counts = [n_chunk * length(basis.krange_allprocs[rank][σ])
                      for rank in 1:mpi_nprocs(basis.comm_kpts)]
            displs = [n_chunk * (first(basis.krange_allprocs[rank][σ])-1)
                      for rank in 1:mpi_nprocs(basis.comm_kpts)]
            @assert all(displs .+ counts .≤ length(data))
            sendbuf = MPI.VBuffer(data, counts, displs)
        else
            sendbuf = nothing
            chunkshape = nothing
        end
        chunkshape = MPI.bcast(chunkshape, 0, basis.comm_kpts)
        destbuf = zeros(T, chunkshape..., length(basis.krange_thisproc[σ]))

        # Scatter and split
        MPI.Scatterv!(sendbuf, destbuf, basis.comm_kpts)
        for (ik, slice) in zip(krange_spin(basis, σ),
                               eachslice(destbuf; dims=ndims(destbuf)))
            splitted[ik] = slice
        end
    end
    if N == 1
        getindex.(splitted)  # Transform Vector{Array{T,0}} => Vector{T}
    else
        splitted
    end
end

"""
Forward FFT calls to the PlaneWaveBasis fft_grid field
"""
ifft!(f_real::AbstractArray3, basis::PlaneWaveBasis, f_fourier::AbstractArray3) = 
    ifft!(f_real, basis.fft_grid, f_fourier)

ifft!(f_real::AbstractArray3, basis::PlaneWaveBasis, kpt::Kpoint, 
      f_fourier::AbstractVector; normalize=true) =
    ifft!(f_real, basis.fft_grid, kpt, f_fourier; normalize=normalize)

ifft(basis::PlaneWaveBasis, f_fourier::AbstractArray) = ifft(basis.fft_grid, f_fourier)

ifft(basis::PlaneWaveBasis, kpt::Kpoint, f_fourier::AbstractVector; kwargs...) = 
    ifft(basis.fft_grid, kpt, f_fourier; kwargs ...)

irfft(basis::PlaneWaveBasis, f_fourier::AbstractArray) = irfft(basis.fft_grid, f_fourier)

fft!(f_fourier::AbstractArray3, basis::PlaneWaveBasis, f_real::AbstractArray3) =
    fft!(f_fourier, basis.fft_grid, f_real)

fft!(f_fourier::AbstractVector, basis::PlaneWaveBasis, kpt::Kpoint, 
     f_real::AbstractArray3; normalize=true) =
    fft!(f_fourier, basis.fft_grid, kpt, f_real; normalize=normalize)

fft(basis::PlaneWaveBasis, f_real::AbstractArray) = fft(basis.fft_grid, f_real)

fft(basis::PlaneWaveBasis, kpt::Kpoint, f_real::AbstractArray3; kwargs...) =
    fft(basis.fft_grid, kpt, f_real; kwargs...)

ifft_matrix(basis::PlaneWaveBasis) = ifft_matrix(basis.fft_grid)
fft_matrix(basis::PlaneWaveBasis) = fft_matrix(basis.fft_grid)
