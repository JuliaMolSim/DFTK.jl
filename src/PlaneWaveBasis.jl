import MPI

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
struct Kpoint{T <: Real, GT <: AbstractVector{Vec3{Int}}}
    spin::Int                     # Spin component can be 1 or 2 as index into what is
    #                             # returned by the `spin_components` function
    coordinate::Vec3{T}           # Fractional coordinate of k-point
    mapping::Vector{Int}          # Index of G_vectors[i] on the FFT grid:
    #                             # G_vectors(basis)[kpt.mapping[i]] == G_vectors(basis, kpt)[i]
    mapping_inv::Dict{Int, Int}   # Inverse of `mapping`:
    #                             # G_vectors(basis)[i] == G_vectors(basis, kpt)[mapping_inv[i]]
    G_vectors::GT                 # Wave vectors in integer coordinates (vector of Vec3{Int})
    #                             # ({G, 1/2 |k+G|^2 ≤ Ecut})
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

`ifft` and `fft` convert between these representations.
"""
struct PlaneWaveBasis{T,
                      VT <: Real,
                      Arch <: AbstractArchitecture,
                      T_G_vectors  <: AbstractArray{Vec3{Int}, 3},
                      T_r_vectors  <: AbstractArray{Vec3{VT},  3},
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

    ## Plans for forward and backward FFT
    # All these plans are *completely unnormalized* (eg FFT * BFFT != I)
    # The normalizations are performed in ifft/fft according to
    # the DFTK conventions (see above)
    opFFT   # out-of-place FFT plan
    ipFFT   # in-place FFT plan
    opBFFT  # inverse plans (unnormalized plan; backward in FFTW terminology)
    ipBFFT
    fft_normalization::T   # fft = fft_normalization * FFT
    ifft_normalization::T  # ifft = ifft_normalization * BFFT

    # "cubic" basis in reciprocal and real space, on which potentials and densities are stored
    G_vectors::T_G_vectors
    r_vectors::T_r_vectors

    ## MPI-local information of the kpoints this processor treats
    # Irreducible kpoints. In the case of collinear spin,
    # this lists all the spin up, then all the spin down
    kpoints::Vector{Kpoint{T, T_kpt_G_vecs}}
    # BZ integration weights, summing up to model.n_spin_components
    kweights::Vector{T}

    ## (MPI-global) information on the k-point grid
    ## These fields are not actually used in computation, but can be used to reconstruct a basis
    # Monkhorst-Pack grid used to generate the k-points, or nothing for custom k-points
    kgrid::Union{Nothing,Vec3{Int}}
    kshift::Union{Nothing,Vec3{T}}
    # full list of (non spin doubled) k-point coordinates in the irreducible BZ
    kcoords_global::Vector{Vec3{T}}
    kweights_global::Vector{T}

    ## Setup for MPI-distributed processing over k-points
    comm_kpts::MPI.Comm           # communicator for the kpoints distribution
    krange_thisproc::Vector{Int}  # indices of kpoints treated explicitly by this
    #                               processor in the global kcoords array
    krange_allprocs::Vector{Vector{Int}}  # indices of kpoints treated by the
    #                                       respective rank in comm_kpts

    ## Information on the hardware and device used for computations.
    architecture::Arch

    ## Symmetry operations that leave the discretized model (k and r grids) invariant.
    # Subset of model.symmetries.
    symmetries::Vector{SymOp{VT}}
    # Whether the symmetry operations leave the rgrid invariant
    # If this is true, the symmetries are a property of the complete discretized model.
    # Therefore, all quantities should be symmetric to machine precision
    symmetries_respect_rgrid::Bool

    ## Instantiated terms (<: Term). See Hamiltonian for high-level usage
    terms::Vector{Any}
end


# prevent broadcast
Base.Broadcast.broadcastable(basis::PlaneWaveBasis) = Ref(basis)

Base.eltype(::PlaneWaveBasis{T}) where {T} = T

@timing function build_kpoints(model::Model{T}, fft_size, kcoords, Ecut;
                               variational=true,
                               architecture::AbstractArchitecture) where {T}
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
            if !variational || norm2(model.recip_lattice * (G + k)) / 2 ≤ Ecut
                push!(mapping, i)
                push!(Gvecs_k, G)
            end
        end
        Gvecs_k = to_device(architecture, Gvecs_k)

        mapping_inv = Dict(ifull => iball for (iball, ifull) in enumerate(mapping))
        for iσ = 1:model.n_spin_components
            push!(kpoints_per_spin[iσ],
                  Kpoint(iσ, k, mapping, mapping_inv, Gvecs_k))
        end
    end
    vcat(kpoints_per_spin...)  # put all spin up first, then all spin down
end
function build_kpoints(basis::PlaneWaveBasis, kcoords)
    build_kpoints(basis.model, basis.fft_size, kcoords, basis.Ecut;
                  variational=basis.variational, basis.architecture)
end

# Lowest-level constructor, should not be called directly.
# All given parameters must be the same on all processors
# and are stored in PlaneWaveBasis for easy reconstruction.
function PlaneWaveBasis(model::Model{T}, Ecut::Number, fft_size, variational,
                        kcoords, kweights, kgrid, kshift,
                        symmetries_respect_rgrid, comm_kpts,
                        architecture::Arch
                       ) where {T <: Real, Arch <: AbstractArchitecture}
    # Validate fft_size
    if variational
        max_E = norm2(model.recip_lattice * floor.(Int, Vec3(fft_size) ./ 2)) / 2
        Ecut > max_E && @warn(
            "For a variational method, Ecut should be less than the maximal kinetic " *
            "energy the grid supports ($max_E)"
        )
    end
    if !(all(fft_size .== next_working_fft_size(T, fft_size)))
        @show fft_size next_working_fft_size(T, fft_size)
        error("Selected fft_size will not work for the buggy generic " *
              "FFT routines; use next_working_fft_size")
    end
    fft_size   = Tuple{Int, Int, Int}(fft_size)  # explicit conversion in case passed as array
    N1, N2, N3 = fft_size

    # filter out the symmetries that don't preserve the real-space grid
    symmetries = model.symmetries
    if symmetries_respect_rgrid
        symmetries = symmetries_preserving_rgrid(symmetries, fft_size)
    end

    # build or validate the kgrid, and get symmetries preserving the kgrid
    if isnothing(kcoords)
        # MP grid based on kgrid/kshift
        @assert !isnothing(kgrid)
        @assert !isnothing(kshift)
        @assert isnothing(kweights)
        kcoords, kweights, symmetries = bzmesh_ir_wedge(kgrid, symmetries; kshift)
    else
        # Manual kpoint set based on kcoords/kweights
        @assert length(kcoords) == length(kweights)
        all_kcoords = unfold_kcoords(kcoords, symmetries)
        symmetries  = symmetries_preserving_kgrid(symmetries, all_kcoords)
    end

    # Init MPI, and store MPI-global values for reference
    MPI.Init()
    kcoords_global  = kcoords
    kweights_global = kweights

    # Setup FFT plans
    Gs = to_device(architecture, G_vectors(fft_size))
    (ipFFT, opFFT, ipBFFT, opBFFT) = build_fft_plans!(similar(Gs, Complex{T}, fft_size))

    # Normalization constants
    # fft = fft_normalization * FFT
    # The convention we want is
    # ψ(r) = sum_G c_G e^iGr / sqrt(Ω)
    # so that the ifft has to normalized by 1/sqrt(Ω).
    # The other constant is chosen because FFT * BFFT = N
    ifft_normalization = 1/sqrt(model.unit_cell_volume)
    fft_normalization  = sqrt(model.unit_cell_volume) / length(ipFFT)

    # Compute k-point information and spread them across processors
    # Right now we split only the kcoords: both spin channels have to be handled
    # by the same process
    n_kpt   = length(kcoords_global)
    n_procs = mpi_nprocs(comm_kpts)

    # Custom reduction operators for MPI are currently not working on aarch64, so
    # fallbacks are defined in common/mpi.jl. For them to work, there cannot be more
    # than 1 MPI process.
    if Base.Sys.ARCH == :aarch64 && n_procs > 1
        error("MPI not supported on aarch64 " *
              "(see https://github.com/JuliaParallel/MPI.jl/issues/404)")
    end

    if n_procs > n_kpt
        # XXX Supporting more processors than kpoints would require
        # fixing a bunch of "reducing over empty collections" errors
        # In the unit tests it is really annoying that this fails so we hack around it, but
        # generally it leads to duplicated work that is not in the users interest.
        if parse(Bool, get(ENV, "CI", "false"))
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
    kweights_thisproc = kweights[krange_thisproc]

    # Setup k-point basis sets
    !variational && @warn(
        "Non-variational calculations are experimental. " *
        "Not all features of DFTK may be supported or work as intended."
    )
    kpoints = build_kpoints(model, fft_size, kcoords_global[krange_thisproc], Ecut;
                            variational, architecture)
    # kpoints is now possibly twice the size of krange. Make things consistent
    if model.n_spin_components == 2
        krange_thisproc   = vcat(krange_thisproc, n_kpt .+ krange_thisproc)
        krange_allprocs   = [vcat(range, n_kpt .+ range) for range in krange_allprocs]
        kweights_thisproc = vcat(kweights_thisproc, kweights_thisproc)
    end
    @assert mpi_sum(sum(kweights_thisproc), comm_kpts) ≈ model.n_spin_components
    @assert length(kpoints) == length(kweights_thisproc)

    if architecture isa GPU && Threads.nthreads() > 1
        error("Can't mix multi-threading and GPU computations yet.")
    end

    VT = value_type(T)
    dvol  = model.unit_cell_volume ./ prod(fft_size)
    r_vectors = [Vec3{VT}(VT(i-1) / N1, VT(j-1) / N2, VT(k-1) / N3)
                 for i = 1:N1, j = 1:N2, k = 1:N3]
    r_vectors = to_device(architecture, r_vectors)
    terms = Vector{Any}(undef, length(model.term_types))  # Dummy terms array, filled below

    basis = PlaneWaveBasis{T, value_type(T), Arch, typeof(Gs), typeof(r_vectors),
                           typeof(kpoints[1].G_vectors)}(
        model, fft_size, dvol,
        Ecut, variational,
        opFFT, ipFFT, opBFFT, ipBFFT,
        fft_normalization, ifft_normalization,
        Gs, r_vectors,
        kpoints, kweights_thisproc, kgrid, kshift,
        kcoords_global, kweights_global, comm_kpts, krange_thisproc, krange_allprocs,
        architecture, symmetries, symmetries_respect_rgrid, terms)
    # Instantiate the terms with the basis
    for (it, t) in enumerate(model.term_types)
        term_name = string(nameof(typeof(t)))
        @timing "Instantiation $term_name" basis.terms[it] = t(basis)
    end
    basis
end

# This is an intermediate-level constructor, which allows for the
# custom specification of k points and G grids.
# For regular usage, the higher-level one below should be preferred
@timing function PlaneWaveBasis(model::Model{T}, Ecut::Number,
                                kcoords ::Union{Nothing, AbstractVector},
                                kweights::Union{Nothing, AbstractVector};
                                variational=true, fft_size=nothing,
                                kgrid=nothing, kshift=nothing,
                                symmetries_respect_rgrid=isnothing(fft_size),
                                comm_kpts=MPI.COMM_WORLD, architecture=CPU()) where {T <: Real}
    if isnothing(fft_size)
        @assert variational
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
            factors = (1,)
        end
        fft_size = compute_fft_size(model, Ecut, kcoords; factors)
    end
    PlaneWaveBasis(model, Ecut, fft_size, variational, kcoords, kweights,
                   kgrid, kshift, symmetries_respect_rgrid, comm_kpts, architecture)
end

@doc raw"""
Creates a `PlaneWaveBasis` using the kinetic energy cutoff `Ecut` and a Monkhorst-Pack
``k``-point grid. The MP grid can either be specified directly with `kgrid` providing the
number of points in each dimension and `kshift` the shift (0 or 1/2 in each direction).
If not specified a grid is generated using `kgrid_from_minimal_spacing` with
a minimal spacing of `2π * 0.022` per Bohr.
"""
function PlaneWaveBasis(model::Model;
                        Ecut,
                        kgrid=kgrid_from_minimal_spacing(model, 2π * 0.022),
                        kshift=zeros(3),
                        kwargs...)
    PlaneWaveBasis(model, austrip(Ecut), nothing, nothing;
                   kgrid, kshift, kwargs...)
end

"""
Creates a new basis identical to `basis`, but with a custom set of kpoints
"""
@timing function PlaneWaveBasis(basis::PlaneWaveBasis, kcoords::AbstractVector,
                                kweights::AbstractVector)
    kgrid = kshift = nothing
    PlaneWaveBasis(basis.model, basis.Ecut,
                   basis.fft_size, basis.variational,
                   kcoords, kweights, kgrid, kshift,
                   basis.symmetries_respect_rgrid, basis.comm_kpts, basis.architecture)
end

"""
    G_vectors([architecture=AbstractArchitecture], fft_size::Tuple)

The wave vectors `G` in reduced (integer) coordinates for a cubic basis set
of given sizes.
"""
function G_vectors(fft_size::Union{Tuple,AbstractVector})
    # Note that a collect(G_vectors_generator(fft_size)) is 100-fold slower
    # than this implementation, hence the code duplication.
    start = .- cld.(fft_size .- 1, 2)
    stop  = fld.(fft_size .- 1, 2)
    axes  = [[collect(0:stop[i]); collect(start[i]:-1)] for i in 1:3]
    [Vec3{Int}(i, j, k) for i in axes[1], j in axes[2], k in axes[3]]
end

function G_vectors_generator(fft_size::Union{Tuple,AbstractVector})
    # The generator version is used mainly in symmetry.jl for lowpass_for_symmetry! and
    # accumulate_over_symmetries!, which are 100-fold slower with G_vector(fft_size).
    start = .- cld.(fft_size .- 1, 2)
    stop  = fld.(fft_size .- 1, 2)
    axes = [[collect(0:stop[i]); collect(start[i]:-1)] for i in 1:3]
    (Vec3{Int}(i, j, k) for i in axes[1], j in axes[2], k in axes[3])
end


@doc raw"""
    G_vectors(basis::PlaneWaveBasis)
    G_vectors(basis::PlaneWaveBasis, kpt::Kpoint)

The list of wave vectors ``G`` in reduced (integer) coordinates of a `basis`
or a ``k``-point `kpt`.
"""
G_vectors(basis::PlaneWaveBasis) = basis.G_vectors
G_vectors(::PlaneWaveBasis, kpt::Kpoint) = kpt.G_vectors



@doc raw"""
    G_vectors_cart(basis::PlaneWaveBasis)
    G_vectors_cart(basis::PlaneWaveBasis, kpt::Kpoint)

The list of ``G`` vectors of a given `basis` or `kpt`, in cartesian coordinates.
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

The list of ``G + k`` vectors, in cartesian coordinates.
"""
function Gplusk_vectors_cart(basis::PlaneWaveBasis, kpt::Kpoint)
    map(recip_vector_red_to_cart(basis.model), Gplusk_vectors(basis, kpt))
end

@doc raw"""
    r_vectors(basis::PlaneWaveBasis)

The list of ``r`` vectors, in reduced coordinates. By convention, this is in [0,1)^3.
"""
r_vectors(basis::PlaneWaveBasis) = basis.r_vectors

@doc raw"""
    r_vectors_cart(basis::PlaneWaveBasis)

The list of ``r`` vectors, in cartesian coordinates.
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

function index_G_vectors(basis::PlaneWaveBasis, G::AbstractVector{<:Integer})
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
    kcoords  = getproperty.(basis.kpoints, :coordinate)
    kcoords  = gather_kpts(kcoords, basis)
    kweights = gather_kpts(basis.kweights, basis)

    # Number of distinct k-point coordinates is number of k-points with spin 1
    n_spinup_thisproc = count(kpt.spin == 1 for kpt in basis.kpoints)
    n_kcoords = mpi_sum(n_spinup_thisproc, basis.comm_kpts)

    if isnothing(kcoords)  # i.e. master process
        nothing
    else
        PlaneWaveBasis(basis.model,
                       basis.Ecut,
                       kcoords[1:n_kcoords],
                       kweights[1:n_kcoords];
                       basis.variational,
                       basis.kgrid,
                       basis.kshift,
                       basis.symmetries_respect_rgrid,
                       comm_kpts=MPI.COMM_SELF,
                       architecture=basis.architecture)
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
            rk_data, status = MPI.recv(basis.comm_kpts, MPI.Status; source=rank, tag=tag)
            @assert MPI.Get_error(status) == 0  # all went well
            allk_data[basis.krange_allprocs[rank + 1]] = rk_data
        end
        allk_data
    else
        MPI.send(data, master, tag, basis.comm_kpts)
        nothing
    end
end
