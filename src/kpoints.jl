"""
Discretization information for ``k``-point-dependent quantities such as orbitals.
More generally, a ``k``-point is a block of the Hamiltonian;
e.g. collinear spin is treated by doubling the number of ``k``-points.
"""
struct Kpoint{T <: Real, GT <: AbstractVector{Vec3{Int}}}
    spin::Int                     # Spin component can be 1 or 2 as index into what is
    #                             # returned by the `spin_components` function
    coordinate::Vec3{T}           # Fractional coordinate of k-point
    G_vectors::GT                 # Wave vectors in integer coordinates (vector of Vec3{Int})
    #                             # ({G, 1/2 |k+G|^2 ≤ Ecut})
                                  # This is not assumed to be in any particular order
    mapping::Vector{Int}          # Index of G_vectors[i] on the FFT grid:
    #                             # G_vectors(basis)[kpt.mapping[i]] == G_vectors(basis, kpt)[i]
    mapping_inv::Dict{Int, Int}   # Inverse of `mapping`:
    #                             # G_vectors(basis)[i] == G_vectors(basis, kpt)[mapping_inv[i]]
end

function Kpoint(spin::Integer, coordinate::AbstractVector{<:Real},
                recip_lattice::AbstractMatrix{T}, fft_size, Ecut;
                variational=true, architecture::AbstractArchitecture) where {T}
    mapping = Int[]
    Gvecs_k = Vec3{Int}[]
    k = Vec3{T}(coordinate)
    # provide a rough hint so that the arrays don't have to be resized so much
    n_guess = div(prod(fft_size), 8)
    sizehint!(mapping, n_guess)
    sizehint!(Gvecs_k, n_guess)
    for (i, G) in enumerate(G_vectors(fft_size))
        if !variational || norm2(recip_lattice * (G + k)) / 2 ≤ Ecut
            push!(mapping, i)
            push!(Gvecs_k, G)
        end
    end
    Gvecs_k = to_device(architecture, Gvecs_k)

    mapping_inv = Dict(ifull => iball for (iball, ifull) in enumerate(mapping))
    Kpoint(spin, k, Gvecs_k, mapping, mapping_inv)
end

# Construct the kpoint with coordinate equivalent_kpt.coordinate + ΔG.
# Equivalent to (but faster than) Kpoint(equivalent_kpt.coordinate + ΔG).
function construct_from_equivalent_kpt(fft_size, equivalent_kpt, coordinate, ΔG)
    linear = LinearIndices(fft_size)
    # Mapping is the same as if created from scratch, although it is not ordered.
    mapping = map(CartesianIndices(fft_size)[equivalent_kpt.mapping]) do G
        linear[CartesianIndex(mod1.(Tuple(G + CartesianIndex(ΔG...)), fft_size))]
    end
    mapping_inv = Dict(ifull => iball for (iball, ifull) in enumerate(mapping))
    Kpoint(equivalent_kpt.spin, Vec3(coordinate), equivalent_kpt.G_vectors .+ Ref(ΔG),
           mapping, mapping_inv)
end

@timing function build_kpoints(model::Model{T}, fft_size, kcoords, Ecut;
                               variational=true,
                               architecture::AbstractArchitecture) where {T}
    # Build all k-points for the first spin
    kpoints_spin_1 = [Kpoint(1, k, model.recip_lattice, fft_size, Ecut;
                             variational, architecture)
                      for k in kcoords]
    all_kpoints = similar(kpoints_spin_1, 0)
    for iσ = 1:model.n_spin_components
        for kpt in kpoints_spin_1
            push!(all_kpoints, Kpoint(iσ, kpt.coordinate,
                                      kpt.G_vectors, kpt.mapping, kpt.mapping_inv))
        end
    end
    all_kpoints
end

# Forward FFT calls taking a Kpoint as argument
ifft!(f_real::AbstractArray3, fft_grid::FFTGrid, kpt::Kpoint,
      f_fourier::AbstractVector; normalize=true) =
    ifft!(f_real, fft_grid, kpt.mapping, f_fourier; normalize=normalize)

ifft(fft_grid::FFTGrid, kpt::Kpoint, f_fourier::AbstractVector; kwargs...) =
    ifft(fft_grid, kpt.mapping, f_fourier; kwargs...)

fft!(f_fourier::AbstractVector, fft_grid::FFTGrid, kpt::Kpoint,
     f_real::AbstractArray3; normalize=true) =
     fft!(f_fourier, fft_grid, kpt.mapping, f_real; normalize=normalize)

fft(fft_grid::FFTGrid, kpt::Kpoint, f_real::AbstractArray3; kwargs...) =
    fft(fft_grid, kpt.mapping, f_real; kwargs...)

### WIP: KpointSet, a data structure that contains a set of k-points and related data

struct KpointSet{T}

    ## MPI-local information of the kpoints this processor treats
    # Irreducible kpoints. In the case of collinear spin,
    # this lists all the spin up, then all the spin down
    kpoints::Vector{Kpoint{T, Vector{Vec3{Int}}}}
    # BZ integration weights, summing up to model.n_spin_components
    kweights::Vector{T}

    ## (MPI-global) information on the k-point grid
    ## These fields are not actually used in computation, but can be used to reconstruct a basis
    # Monkhorst-Pack grid used to generate the k-points, or nothing for custom k-points
    kgrid::AbstractKgrid
    # full list of (non spin doubled) k-point coordinates in the irreducible BZ
    kcoords_global::Vector{Vec3{T}}
    kweights_global::Vector{T}

    # Number of irreducible k-points in the set. If there are more MPI ranks than irreducible
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

end

function KpointSet(model::Model{T}, Ecut::Real, fft_size::Tuple{Int, Int, Int},
                   variational::Bool, kgrid::AbstractKgrid,
                   symmetries::AbstractVector,
                   use_symmetries_for_kpoint_reduction::Bool,
                   comm_kpts, architecture::Arch
                   ) where {T <: Real, Arch <: AbstractArchitecture}
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

    KpointSet{T}(
        kpoints, kweights, kgrid, kcoords_global, kweights_global,
        n_irreducible_kpoints, comm_kpts, krange_thisproc,
        krange_allprocs, krange_thisproc_allspin
    )
end

"""
Utilities to get information about the irreducible k-point mesh (in case of duplication)
Useful for I/O, where k-point information should not be duplicated
"""
function irreducible_kcoords_global(kpoint_set::KpointSet)
    # Assume that duplicated k-points are appended at the end of the kcoords array
    kpoint_set.kcoords_global[1:kpoint_set.n_irreducible_kpoints]
end

function irreducible_kweights_global(kpoint_set::KpointSet)
    function same_kpoint(i_irr, i_dupl)
        maximum(abs, kpoint_set.kcoords_global[i_dupl]-kpoint_set.kcoords_global[i_irr]) ≈ 0
    end

    # Check that weights add up to 1 on entry (non spin doubled k-points)
    @assert sum(kpoint_set.kweights_global) ≈ 1

    # Assume that duplicated k-points are appended at the end of the kcoords array
    irr_kweights = kpoint_set.kweights_global[1:kpoint_set.n_irreducible_kpoints]
    for i_dupl = kpoint_set.n_irreducible_kpoints+1:length(kpoint_set.kweights_global)
        for i_irr = 1:kpoint_set.n_irreducible_kpoints
            if same_kpoint(i_irr, i_dupl)
                irr_kweights[i_irr] += kpoint_set.kweights_global[i_dupl]
                break
            end
        end
    end

    # Test that irreducible weight add up to 1 (non spin doubled k-points)
    @assert sum(irr_kweights) ≈ 1
    irr_kweights
end

"""
Sum an array over kpoints, taking weights into account
"""
function weighted_ksum(kpoint_set::KpointSet, array)
    res = sum(kpoint_set.kweights .* array)
    mpi_sum(res, kpoint_set.comm_kpts)
end