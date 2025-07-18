import Spglib
import Brillouin.KPaths: KPathInterpolant

"""Bring ``k``-point coordinates into the range [-0.5, 0.5)"""
function normalize_kpoint_coordinate(x::Real)
    x = x - round(Int, x, RoundNearestTiesUp)
    @assert -0.5 ≤ x < 0.5
    x
end
normalize_kpoint_coordinate(k::AbstractVector) = normalize_kpoint_coordinate.(k)


# A datastructure that can produce a full list of k-points (reducible_kcoords)
# as well as a symmetry-reduced list of k-points (irreducible_kcoords)
abstract type AbstractKgrid end

"""
Perform BZ sampling employing a Monkhorst-Pack grid.

See also [`KgridSpacing`](@ref) and [`KgridTotalNumber`](@ref)
for ways to set up k-point grids dynamically in coordination with the atomistic
structure of the calculation.
"""
struct MonkhorstPack <: AbstractKgrid
    kgrid_size::Vec3{Int}
    kshift::Vec3{Rational{Int}}
end
MonkhorstPack(kgrid_size::AbstractVector; kshift=[0, 0, 0]) = MonkhorstPack(kgrid_size, kshift)
MonkhorstPack(kgrid_size::Tuple; kshift=[0, 0, 0]) = MonkhorstPack(kgrid_size, kshift)
MonkhorstPack(k1::Integer, k2::Integer, k3::Integer) = MonkhorstPack([k1, k2, k3])
function Base.show(io::IO, kgrid::MonkhorstPack)
    print(io, "MonkhorstPack(", kgrid.kgrid_size)
    if !iszero(kgrid.kshift)
        print(io, ", ", Float64.(kgrid.kshift))
    end
    print(io, ")")
end
Base.length(kgrid::MonkhorstPack) = prod(kgrid.kgrid_size)

"""Construct the coordinates of the k-points in a (shifted) Monkhorst-Pack grid"""
function reducible_kcoords(kgrid::MonkhorstPack)
    kgrid_size = kgrid.kgrid_size
    start   = -floor.(Int, (kgrid_size .- 1) ./ 2)
    stop    = ceil.(Int, (kgrid_size .- 1) ./ 2)
    kcoords = [(kgrid.kshift .+ Vec3([i, j, k])) .// kgrid_size
               for i=start[1]:stop[1], j=start[2]:stop[2], k=start[3]:stop[3]]
    (; kcoords=vec(normalize_kpoint_coordinate.(kcoords)))
end

"""
Construct the irreducible wedge given the crystal `symmetries`. Returns the list of k-point
coordinates and the associated weights.
"""
function irreducible_kcoords(kgrid::MonkhorstPack, symmetries::AbstractVector{<:SymOp};
                             check_symmetry=SYMMETRY_CHECK)
    if all(isone, kgrid.kgrid_size)
        return (; kcoords=[Vec3{Float64}(kgrid.kshift)], kweights=[1.0])
    end

    # Give the remaining symmetries to spglib to compute an irreducible k-point mesh
    # TODO implement time-reversal symmetry and turn the flag below to true
    is_shift = map(kgrid.kshift) do ks
        ks in (0, 1//2) || error("Only kshifts of 0 or 1//2 implemented.")
        ks == 1//2
    end
    rotations = [symop.W for symop in symmetries]
    qpoints = [Vec3(0, 0, 0)]
    spg_mesh = Spglib.get_stabilized_reciprocal_mesh(rotations, kgrid.kgrid_size, qpoints;
                                                     is_shift, is_time_reversal=false)
    kirreds = map(Spglib.eachpoint(spg_mesh)) do kcoord
        normalize_kpoint_coordinate(Vec3(kcoord))
    end

    # Find the indices of the corresponding reducible k-points in the MP grid, which one of
    # the irreducible k-points in `kirreds` generates.
    ir_mapping = spg_mesh.ir_mapping_table
    k_all_reducible = [findall(isequal(elem), ir_mapping) for elem in unique(ir_mapping)]

    # Number of reducible k-points represented by the irreducible k-point `kirreds[ik]`
    n_equivalent_k = length.(k_all_reducible)
    @assert sum(n_equivalent_k) == prod(kgrid.kgrid_size)
    kweights = n_equivalent_k / sum(n_equivalent_k)

    # This loop checks for reducible k-points, which could not be mapped to any irreducible
    # k-point yet even though spglib claims this can be done.
    # This happens because spglib actually fails for some non-ideal lattices, resulting
    # in *wrong results* being returned. See the discussion in
    # https://github.com/spglib/spglib/issues/101
    if check_symmetry
        _check_kpoint_reduction(symmetries, kgrid, k_all_reducible,
                                kirreds, spg_mesh.grid_address)
    end

    (; kcoords=kirreds, kweights)
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
function ExplicitKpoints(kinter::KPathInterpolant{D}) where {D}
    ExplicitKpoints(map(k -> vcat(k, zeros_like(k, 3 - D)), kinter))
end
Base.convert(::Type{<:AbstractKgrid}, kinter::KPathInterpolant) = ExplicitKpoints(kinter)

function Base.show(io::IO, kgrid::ExplicitKpoints)
    print(io, "ExplicitKpoints with $(length(kgrid.kcoords)) k-points")
end
Base.length(kgrid::ExplicitKpoints) = length(kgrid.kcoords)
reducible_kcoords(kgrid::ExplicitKpoints) = (; kgrid.kcoords)
function irreducible_kcoords(kgrid::ExplicitKpoints, ::AbstractVector{<:SymOp})
    (; kgrid.kcoords, kgrid.kweights)
end


# A datastructure that should return an AbstractKgrid (usually a MonkhorstPack)
# when used with `build_kgrid`.
abstract type  AbstractKgridGenerator end

build_kgrid(lattice::AbstractMatrix, kgrid::AbstractKgrid) = kgrid
function build_kgrid(lattice::AbstractMatrix, kgrid::Union{AbstractVector,Tuple})
    build_kgrid(lattice, MonkhorstPack(kgrid; kshift=[0, 0, 0]))
end

struct KgridSpacing <: AbstractKgridGenerator
    spacing::Float64
    kshift::Vec3{Rational{Int}}
end
function Base.show(io::IO, kgen::KgridSpacing)
    print(io, "KgridSpacing($(kgen.spacing) / u\"bohr\"")
    if !iszero(kgen.kshift)
        print(io, ", ", Float64.(kgen.kshift))
    end
    print(io, ")")
end

@doc raw"""
Request a [`MonkhorstPack`](@ref) grid to ensure kpoints are at most this `spacing`
apart (in inverse Bohrs). A reasonable spacing is `0.13` inverse Bohrs
(around ``2π * 0.04 \, \text{Å}^{-1}``). The `kshift` keyword argument allows
to specify an explicit shift for all ``k``-points.

Internally the `MonkhorstPack` object will be generated by calling
`build_kgrid(lattice, kgen)` where `kgen` is this object.
"""
function KgridSpacing(spacing; kshift=[0, 0, 0])
    KgridSpacing(austrip(spacing), rationalize.(kshift; tol=1e-6))
end
KgridSpacing(spacing::Quantity, kshift) = KgridSpacing(austrip(spacing); kshift)

function build_kgrid(lattice::AbstractMatrix, kgen::KgridSpacing)
    lattice = austrip.(lattice)
    recip_lattice = compute_recip_lattice(lattice)
    isinf(kgen.spacing) && return MonkhorstPack([1, 1, 1], kgen.kshift)
    kgrid = [max(1, ceil(Int, norm(vec) / kgen.spacing)) for vec in eachcol(recip_lattice)]
    MonkhorstPack(kgrid, kgen.kshift)
end

@doc raw"""
Selects a [`MonkhorstPack`](@ref) grid size which ensures that at least a
`n_kpoints` total number of ``k``-points are used. The distribution of
``k``-points amongst coordinate directions is as uniformly as possible, trying to
achieve an identical minimal spacing in all directions.

Internally the `MonkhorstPack` object will be generated by calling
`build_kgrid(lattice, kgen)` where `kgen` is this object.
"""
struct KgridTotalNumber <: AbstractKgridGenerator
    n_kpoints::Int
    kshift::Vec3{Rational{Int}}
end
function KgridTotalNumber(n_kpoints::Integer; kshift=[0, 0, 0])
    if n_kpoints ≤ 0
        throw(ArgumentError("n_kpoints should be a positive number"))
    end
    KgridTotalNumber(n_kpoints, rationalize.(kshift; tol=1e-6))
end
function Base.show(io::IO, kgen::KgridTotalNumber)
    print(io, "KgridTotalNumber(", kgen.n_kpoints)
    if !iszero(kgen.kshift)
        print(io, ", ", Float64.(kgen.kshift))
    end
    print(io, ")")
end

function build_kgrid(lattice::AbstractMatrix, kgen::KgridTotalNumber)
    lattice = austrip.(lattice)
    n_dim   = count(!iszero, eachcol(lattice))
    kgen.n_kpoints ≤ 1 && return MonkhorstPack([1, 1, 1], kgen.kshift)
    n_dim == 1         && return MonkhorstPack([kgen.n_kpoints, 1, 1], kgen.kshift)

    # Compute truncated reciprocal lattice
    recip_lattice_nD = 2π * inv(lattice[1:n_dim, 1:n_dim]')
    n_kpt_per_dim = kgen.n_kpoints^(1/n_dim)

    # Start from a cubic lattice. If it is one, we are done. Otherwise the resulting
    # spacings in each dimension bracket the ideal k-point spacing.
    spacing_per_dim = [norm(vec) / n_kpt_per_dim for vec in eachcol(recip_lattice_nD)]
    min_spacing, max_spacing = extrema(spacing_per_dim)
    if min_spacing ≈ max_spacing
        return build_kgrid(lattice, KgridSpacing(min_spacing; kgen.kshift))
    else
        number_of_kpoints(spacing) = prod(vec -> norm(vec) / spacing, eachcol(recip_lattice_nD))
        @assert number_of_kpoints(min_spacing) + 0.05 ≥ kgen.n_kpoints
        @assert number_of_kpoints(max_spacing) - 0.05 ≤ kgen.n_kpoints

        # TODO This is not optimal and sometimes finds too large grids
        spacing = Roots.find_zero(sp -> number_of_kpoints(sp) - kgen.n_kpoints,
                                  (min_spacing, max_spacing), Roots.Bisection(),
                                  xatol=1e-4, atol=0, rtol=0)

        # Sanity check: Sometimes root finding is just across the edge towards
        # a larger number of k-points than needed. This attempts a slightly larger spacing.
        kgrid_larger = build_kgrid(lattice, KgridSpacing(spacing + 1e-4; kgen.kshift))
        if length(kgrid_larger) ≥ kgen.n_kpoints
            return kgrid_larger
        else
            return build_kgrid(lattice, KgridSpacing(spacing; kgen.kshift))
        end
    end
end


@doc raw"""
Build a [`MonkhorstPack`](@ref) grid to ensure kpoints are at most this `spacing`
apart (in inverse Bohrs). A reasonable spacing is `0.13` inverse Bohrs
(around ``2π * 0.04 \, \text{Å}^{-1}``). The `kshift` keyword argument allows
to specify an explicit shift for all ``k``-points.

Note that this function is deprecated. Use [`KgridSpacing`](@ref) instead.
"""
function kgrid_from_maximal_spacing(system::AbstractSystem, spacing; kshift=[0, 0, 0])
    @warn("kgrid_from_maximal_spacing(system, spacing; kshift) is now deprecated. " *
          "Now you should pass the `KgridSpacing(spacing; kshift)` directly " *
          "as `PlaneWaveBasis(model; kgrid=KgridSpacing(spacing; kshift), ...)` or " *
          "use `build_kgrid(lattice, KgridSpacing(spacing; kshift))`")
    pseudopotentials = fill(nothing, length(system))
    kgen = KgridSpacing(spacing, kshift)
    build_kgrid(parse_system(system, pseudopotentials).lattice, kgen)
end
function kgrid_from_maximal_spacing(lattice::AbstractMatrix, spacing; kshift=[0, 0, 0])
    @warn("kgrid_from_maximal_spacing(system, spacing; kshift) is now deprecated. " *
          "Now you should pass the `KgridSpacing(spacing; kshift)` directly " *
          "as `PlaneWaveBasis(model; kgrid=KgridSpacing(spacing; kshift), ...)` or " *
          "use `build_kgrid(lattice, KgridSpacing(spacing; kshift))`")
    build_kgrid(lattice, KgridSpacing(spacing, kshift))
end

@doc raw"""
Selects a [`MonkhorstPack`](@ref) grid size which ensures that at least a
`n_kpoints` total number of ``k``-points are used. The distribution of
``k``-points amongst coordinate directions is as uniformly as possible, trying to
achieve an identical minimal spacing in all directions.

Note that this function is deprecated. Use [`KgridTotalNumber`](@ref) instead.
"""
function kgrid_from_minimal_n_kpoints(system::AbstractSystem, n_kpoints::Integer; kshift=[0, 0, 0])
    @warn("kgrid_from_minimial_n_kpoints(system, n_kpoints; kshift) is now deprecated. " *
          "Now you should pass the `KgridTotalNumber(n_kpoints; kshift)` directly " *
          "as `PlaneWaveBasis(model; kgrid=KgridTotalNumber(n_kpoints; kshift), ...)` or " *
          "use `build_kgrid(lattice, KgridTotalNumber(n_kpoints; kshift))`")
    pseudopotentials = fill(nothing, length(system))
    kgrid_from_minimal_n_kpoints(parse_system(system, pseudopotentials).lattice, n_kpoints; kshift)
end
function kgrid_from_minimal_n_kpoints(lattice, n_kpoints::Integer; kshift=[0, 0, 0])
    build_kgrid(lattice, KgridTotalNumber(n_kpoints, kshift))
end


@timing function _check_kpoint_reduction(symmetries::AbstractVector{<: SymOp},
                                         kgrid::MonkhorstPack, k_all_reducible, kirreds,
                                         grid_address)
    for (iks_reducible, k) in zip(k_all_reducible, kirreds), ikred in iks_reducible
        kred = (kgrid.kshift .+ grid_address[ikred]) .// kgrid.kgrid_size
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
end
