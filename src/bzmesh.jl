import Spglib

"""Bring ``k``-point coordinates into the range [-0.5, 0.5)"""
function normalize_kpoint_coordinate(x::Real)
    x = x - round(Int, x, RoundNearestTiesUp)
    @assert -0.5 ≤ x < 0.5
    x
end
normalize_kpoint_coordinate(k::AbstractVector) = normalize_kpoint_coordinate.(k)


"""Construct the coordinates of the ``k``-points in a (shifted) Monkorst-Pack grid"""
function kcoords_monkhorst_pack(kgrid_size; kshift=[0, 0, 0])
    kgrid_size = Vec3{Int}(kgrid_size)
    kshift     = Vec3{Rational{Int}}(kshift)

    start   = -floor.(Int, (kgrid_size .- 1) ./ 2)
    stop    = ceil.(Int, (kgrid_size .- 1) ./ 2)
    kcoords = [(kshift .+ Vec3([i, j, k])) .// kgrid_size
               for i=start[1]:stop[1], j=start[2]:stop[2], k=start[3]:stop[3]]
    vec(normalize_kpoint_coordinate.(kcoords))
end


@doc raw"""
    bzmesh_uniform(kgrid_size; kshift=[0, 0, 0])

Construct a (shifted) uniform Brillouin zone mesh for sampling the ``k``-points.
Returns all ``k``-point coordinates, appropriate weights and the identity SymOp.
"""
function bzmesh_uniform(kgrid_size; kshift=[0, 0, 0])
    kcoords = kcoords_monkhorst_pack(kgrid_size; kshift)
    (; kcoords, kweights=ones(length(kcoords)) ./ length(kcoords), symmetries=[one(SymOp)])
end


@doc raw"""
     bzmesh_ir_wedge(kgrid_size, symmetries; kshift=[0, 0, 0])

Construct the irreducible wedge of a uniform Brillouin zone mesh for sampling ``k``-points,
given the crystal symmetries `symmetries`. Returns the list of irreducible ``k``-point
(fractional) coordinates, the associated weights and the new `symmetries` compatible with
the grid.
"""
function bzmesh_ir_wedge(kgrid_size, symmetries; kshift=[0, 0, 0])
    all(isone, kgrid_size) && return bzmesh_uniform(kgrid_size; kshift)
    kgrid_size = Vec3{Int}(kgrid_size)
    kshift = Vec3{Rational{Int}}(kshift)

    # Give the remaining symmetries to spglib to compute an irreducible k-point mesh
    # TODO implement time-reversal symmetry and turn the flag below to true
    is_shift = map(kshift) do ks
        ks in (0, 1//2) || error("Only kshifts of 0 or 1//2 implemented.")
        ks == 1//2
    end
    rotations = [symop.W for symop in symmetries]
    qpoints = [Vec3(0, 0, 0)]
    spg_mesh = Spglib.get_stabilized_reciprocal_mesh(rotations, kgrid_size, qpoints;
                                                     is_shift, is_time_reversal=false)
    kirreds = normalize_kpoint_coordinate.(Spglib.eachpoint(spg_mesh))

    # Find the indices of the corresponding reducible k-points in the MP grid, which one of
    # the irreducible k-points in `kirreds` generates.
    ir_mapping = spg_mesh.ir_mapping_table
    k_all_reducible = [findall(isequal(elem), ir_mapping) for elem in unique(ir_mapping)]

    # Number of reducible k-points represented by the irreducible k-point `kirreds[ik]`
    n_equivalent_k = length.(k_all_reducible)
    @assert sum(n_equivalent_k) == prod(kgrid_size)
    kweights = n_equivalent_k / sum(n_equivalent_k)

    # This loop checks for reducible k-points, which could not be mapped to any irreducible
    # k-point yet even though spglib claims this can be done.
    # This happens because spglib actually fails for some non-ideal lattices, resulting
    # in *wrong results* being returned. See the discussion in
    # https://github.com/spglib/spglib/issues/101
    for (iks_reducible, k) in zip(k_all_reducible, kirreds), ikred in iks_reducible
        grid = spg_mesh.grid_address
        kred = (kshift .+ grid[ikred]) .// kgrid_size
        found_mapping = any(symmetries) do symop
            # If the difference between kred and W' * k == W^{-1} * k
            # is only integer in fractional reciprocal-space coordinates, then
            # kred and S' * k are equivalent k-points
            all(isinteger, kred - (symop.S * k))
        end
        if !found_mapping
            error("The reducible k-point $kred could not be generated from " *
                  "the irreducible kpoints. This points to a bug in spglib.")
        end
    end

    (; kcoords=kirreds, kweights)
end

abstract type AbstractKgrid end

"""
Perform BZ sampling employing a Monkhorst-Pack grid.
"""
struct MonkhorstPack <: AbstractKgrid
    kgrid::Vec3{Int}
    kshift::Vec3{Rational{Int}}
end
MonkhorstPack(kgrid; kshift=[0, 0, 0]) = MonkhorstPack(kgrid, kshift)
function Base.show(io::IO, kgrid::MonkhorstPack)
    print(io, "MonkhorstPack(", kgrid.kgrid)
    if !iszero(kgrid.kshift)
        print(io, ", ", Float64.(kgrid.kshift))
    end
    print(io, ")")
end
Base.length(kgrid::MonkhorstPack) = prod(kgrid.kgrid)
function reducible_kcoords(kgrid::MonkhorstPack)
    # TODO inline function
    (; kcoords=kcoords_monkhorst_pack(kgrid.kgrid; kgrid.kshift))
end
function irreducible_kcoords(kgrid::MonkhorstPack, symmetries::AbstractVector{<:SymOp})
    # TODO Inline function
    bzmesh_ir_wedge(kgrid.kgrid, symmetries; kgrid.kshift)
end


"""
Explicitly define the k-points along which to perform BZ sampling.
(Useful for bandstructure calculations)
"""
struct ExplicitKpoints{T} <: AbstractKgrid
    kcoords::Vector{Vec3{T}}
    kweights::Vector{T}

    function ExplicitKpoints(kcoords::AbstractVector{<:AbstractVector{T}},
                             kweights::AbstractVector{T}) where {T}
        @assert length(kcoords) == length(kweights)
        new{T}(kcoords, kweights)
    end
end
function ExplicitKpoints(kcoords::AbstractVector{<:AbstractVector{T}}) where {T}
    ExplicitKpoints(kcoords, ones(T, length(kcoords)) ./ length(kcoords))
end
function Base.show(io::IO, kgrid::ExplicitKpoints)
    print(io, "ExplicitKpoints with $(length(kgrid.kcoords)) k-points")
end
Base.length(kgrid::ExplicitKpoints) = length(kgrid.kcoords)
reducible_kcoords(kgrid::ExplicitKpoints) = (; kgrid.kcoords)
function irreducible_kcoords(kgrid::ExplicitKpoints, ::AbstractVector{<:SymOp})
    (; kgrid.kcoords, kgrid.kweights)
end


@doc raw"""
Build a [`MonkhorsPack`](@ref) grid to ensure kpoints are at most this spacing
apart (in inverse Bohrs). A reasonable spacing is `0.13` inverse Bohrs
(around ``2π * 0.04 \AA^{-1}``).
"""
function kgrid_from_maximal_spacing(lattice, spacing; kshift=[0, 0, 0])
    lattice       = austrip.(lattice)
    spacing       = austrip(spacing)
    recip_lattice = compute_recip_lattice(lattice)
    @assert spacing > 0
    isinf(spacing) && return [1, 1, 1]

    kgrid = [max(1, ceil(Int, norm(vec) / spacing)) for vec in eachcol(recip_lattice)]
    MonkhorstPack(kgrid, kshift)
end
function kgrid_from_maximal_spacing(model::Model, args...; kwargs...)
    kgrid_from_maximal_spacing(model.lattice, args...; kwargs...)
end
@deprecate kgrid_from_minimal_spacing kgrid_from_maximal_spacing

@doc raw"""
Selects a [`MonkhorsPack`](@ref) grid size which ensures that at least a
`n_kpoints` total number of ``k``-points are used. The distribution of
``k``-points amongst coordinate directions is as uniformly as possible, trying to
achieve an identical minimal spacing in all directions.
"""
function kgrid_from_minimal_n_kpoints(lattice, n_kpoints::Integer; kshift=[0, 0, 0])
    lattice = austrip.(lattice)
    n_dim   = count(!iszero, eachcol(lattice))
    @assert n_kpoints > 0
    n_kpoints == 1 && return MonkhorstPack([1, 1, 1], kshift)
    n_dim == 1     && return MonkhorstPack([n_kpoints, 1, 1], kshift)

    # Compute truncated reciprocal lattice
    recip_lattice_nD = 2π * inv(lattice[1:n_dim, 1:n_dim]')
    n_kpt_per_dim = n_kpoints^(1/n_dim)

    # Start from a cubic lattice. If it is one, we are done. Otherwise the resulting
    # spacings in each dimension bracket the ideal k-point spacing.
    spacing_per_dim = [norm(vec) / n_kpt_per_dim for vec in eachcol(recip_lattice_nD)]
    min_spacing, max_spacing = extrema(spacing_per_dim)
    if min_spacing ≈ max_spacing
        return kgrid_from_maximal_spacing(lattice, min_spacing)
    else
        number_of_kpoints(spacing) = prod(vec -> norm(vec) / spacing, eachcol(recip_lattice_nD))
        @assert number_of_kpoints(min_spacing) + 0.05 ≥ n_kpoints
        @assert number_of_kpoints(max_spacing) - 0.05 ≤ n_kpoints

        # TODO This is not optimal and sometimes finds too large grids
        spacing = Roots.find_zero(sp -> number_of_kpoints(sp) - n_kpoints,
                                  (min_spacing, max_spacing), Roots.Bisection(),
                                  xatol=1e-4, atol=0, rtol=0)

        # Sanity check: Sometimes root finding is just across the edge towards
        # a larger number of k-points than needed. This attempts a slightly larger spacing.
        kgrid_larger = kgrid_from_maximal_spacing(lattice, spacing + 1e-4)
        if prod(kgrid_larger.kgrid) ≥ n_kpoints
            return kgrid_larger
        else
            return kgrid_from_maximal_spacing(lattice, spacing)
        end
    end
end
function kgrid_from_minimal_n_kpoints(model::Model, n_kpoints::Integer; kwargs...)
    kgrid_from_minimal_n_kpoints(model.lattice, n_kpoints; kwargs...)
end
