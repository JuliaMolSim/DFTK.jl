include("external/spglib.jl")


"""Bring ``k``-point coordinates into the range [-0.5, 0.5)"""
function normalize_kpoint_coordinate(x::Real)
    x = x - round(Int, x, RoundNearestTiesUp)
    @assert -0.5 ≤ x < 0.5
    x
end
normalize_kpoint_coordinate(k::AbstractVector) = normalize_kpoint_coordinate.(k)


"""Construct the coordinates of the ``k``-points in a (shifted) Monkorst-Pack grid"""
function kgrid_monkhorst_pack(kgrid_size; kshift=[0, 0, 0])
    kgrid_size = Vec3{Int}(kgrid_size)
    start = -floor.(Int, (kgrid_size .- 1) ./ 2)
    stop  = ceil.(Int, (kgrid_size .- 1) ./ 2)
    kshift = Vec3{Rational{Int}}(kshift)
    kcoords = [(kshift .+ Vec3([i, j, k])) .// kgrid_size
               for i=start[1]:stop[1], j=start[2]:stop[2], k=start[3]:stop[3]]
    vec(normalize_kpoint_coordinate.(kcoords))
end


@doc raw"""
    bzmesh_uniform(kgrid_size; kshift=[0, 0, 0])

Construct a (shifted) uniform Brillouin zone mesh for sampling the ``k``-points.
The function returns a tuple `(kcoords, ksymops)`, where `kcoords` are the list
of ``k``-points and `ksymops` are a list of symmetry operations (for interface
compatibility with `PlaneWaveBasis` and `bzmesh_irreducible`. No symmetry
reduction is attempted, such that there will be `prod(kgrid_size)` ``k``-points
returned and all symmetry operations are the identity.
"""
function bzmesh_uniform(kgrid_size; kshift=[0, 0, 0])
    kcoords = kgrid_monkhorst_pack(kgrid_size; kshift=kshift)
    kcoords, [[identity_symop()] for _ in 1:length(kcoords)], [identity_symop()]
end


@doc raw"""
     bzmesh_ir_wedge(kgrid_size, symmetries; kshift=[0, 0, 0])

Construct the irreducible wedge of a uniform Brillouin zone mesh for sampling ``k``-points.
The function returns a tuple `(kcoords, ksymops)`, where `kcoords` are the list of
irreducible ``k``-points and `ksymops` are a list of symmetry operations for regenerating
the full mesh. `symmetries` is the tuple returned from
`symmetry_operations(lattice, atoms, magnetic_moments)`.
`tol_symmetry` is the tolerance used for searching for symmetry operations.
"""
function bzmesh_ir_wedge(kgrid_size, symmetries; kshift=[0, 0, 0])
    all(isequal.(kgrid_size, 1)) && return bzmesh_uniform(kgrid_size; kshift)

    # Transform kshift to the convention used in spglib:
    #    If is_shift is set (i.e. integer 1), then a shift of 0.5 is performed,
    #    else no shift is performed along an axis.
    kshift = Vec3{Rational{Int}}(kshift)
    all(ks in (0, 1//2) for ks in kshift) || error("Only kshifts of 0 or 1//2 implemented.")

    kpoints_mp = kgrid_monkhorst_pack(kgrid_size, kshift=kshift)

    # Filter those symmetry operations (S,τ) that preserve the MP grid
    symmetries = symmetries_preserving_kgrid(symmetries, kpoints_mp)

    # Give the remaining symmetries to spglib to compute an irreducible k-point mesh
    # TODO implement time-reversal symmetry and turn the flag to true
    is_shift = Int.(2 * kshift)
    Stildes = [S' for (S, τ) in symmetries]
    _, mapping, grid = spglib_get_stabilized_reciprocal_mesh(
        kgrid_size, Stildes, is_shift=is_shift, is_time_reversal=false
    )
    # Convert irreducible k-points to DFTK conventions
    kgrid_size = Vec3{Int}(kgrid_size)
    kirreds = [(kshift .+ grid[ik + 1]) .// kgrid_size
               for ik in unique(mapping)]
    kirreds = normalize_kpoint_coordinate.(kirreds)

    # Find the indices of the corresponding reducible k-points in `grid`, which one of the
    # irreducible k-points in `kirreds` generates.
    k_all_reducible = [findall(isequal(elem), mapping) for elem in unique(mapping)]

    # This list collects a list of extra reducible k-points, which could not be
    # mapped to any irreducible k-point yet even though spglib claims this can be done.
    # This happens because spglib actually fails for non-ideal cases, resulting
    # in *wrong results* being returned. See the discussion in
    # https://github.com/spglib/spglib/issues/101
    kreds_notmapped = empty(kirreds)

    # ksymops will be the list of symmetry operations (for each irreducible k-point)
    # needed to do the respective mapping irreducible -> reducible to get the respective
    # entry in `k_all_reducible`.
    ksymops = Vector{Vector{SymOp}}(undef, length(kirreds))
    for (ik, k) in enumerate(kirreds)
        ksymops[ik] = Vector{SymOp}()
        for ired in k_all_reducible[ik]
            kred = (kshift .+ grid[ired]) .// kgrid_size

            # Note that this relies on the identity coming up first in symmetries
            isym = findfirst(symmetries) do symop
                # If the difference between kred and Stilde' * k == Stilde^{-1} * k
                # is only integer in fractional reciprocal-space coordinates, then
                # kred and S' * k are equivalent k-points
                S = symop[1]
                all(isinteger, kred - (S * k))
            end

            if isym === nothing  # No symop found for $k -> $kred
                push!(kreds_notmapped, normalize_kpoint_coordinate(kred))
            else
                push!(ksymops[ik], symmetries[isym])
            end
        end
    end

    if !isempty(kreds_notmapped)
        # add them as reducible anyway
        Stildes = [S' for (S, τ) in symmetries]
        τtildes = [-S' * τ for (S, τ) in symmetries]
        eirreds, esymops = find_irreducible_kpoints(kreds_notmapped, Stildes, τtildes)
        @info("$(length(kreds_notmapped)) reducible kpoints could not be generated from " *
              "the irreducible kpoints returned by spglib. $(length(eirreds)) of " *
              "these are added as extra irreducible kpoints.")

        append!(kirreds, eirreds)
        append!(ksymops, esymops)
    end

    # The symmetry operation (S == I and τ == 0) should be present for each k-point
    @assert all(findfirst(Sτ -> iszero(Sτ[1] - I) && iszero(Sτ[2]), ops) !== nothing
                for ops in ksymops)

    kirreds, ksymops, symmetries
end


@doc raw"""
Apply various standardisations to a lattice and a list of atoms. It uses spglib to detect
symmetries (within `tol_symmetry`), then cleans up the lattice according to the symmetries
(unless `correct_symmetry` is `false`) and returns the resulting standard lattice
and atoms. If `primitive` is `true` (default) the primitive unit cell is returned, else
the conventional unit cell is returned.
"""
const standardize_atoms = spglib_standardize_cell

# TODO Maybe maximal spacing is actually a better name as the kpoints are spaced
#      at most that far apart
@doc raw"""
Selects a kgrid size to ensure a minimal spacing (in inverse Bohrs) between kpoints.
A reasonable spacing is `0.13` inverse Bohrs (around ``2π * 0.04 \AA^{-1}``).
"""
function kgrid_from_minimal_spacing(lattice, spacing)
    lattice       = austrip.(lattice)
    spacing       = austrip(spacing)
    recip_lattice = compute_recip_lattice(lattice)
    @assert spacing > 0
    isinf(spacing) && return [1, 1, 1]

    [max(1, ceil(Int, norm(vec) / spacing)) for vec in eachcol(recip_lattice)]
end
function kgrid_from_minimal_spacing(model::Model, args...)
    kgrid_from_minimal_spacing(model.lattice, args...)
end

@doc raw"""
Selects a kgrid size which ensures that at least a `n_kpoints` total number of ``k``-points
are used. The distribution of ``k``-points amongst coordinate directions is as uniformly
as possible, trying to achieve an identical minimal spacing in all directions.
"""
function kgrid_from_minimal_n_kpoints(lattice, n_kpoints::Integer)
    lattice = austrip.(lattice)
    n_dim   = count(!iszero, eachcol(lattice))
    @assert n_kpoints > 0
    n_kpoints == 1 && return [1, 1, 1]
    n_dim == 1 && return [n_kpoints, 1, 1]

    # Compute truncated reciprocal lattice
    recip_lattice_nD = 2π * inv(lattice[1:n_dim, 1:n_dim]')
    n_kpt_per_dim = n_kpoints^(1/n_dim)

    # Start from a cubic lattice. If it is one, we are done. Otherwise the resulting
    # spacings in each dimension bracket the ideal k-point spacing.
    spacing_per_dim = [norm(vec) / n_kpt_per_dim for vec in eachcol(recip_lattice_nD)]
    min_spacing, max_spacing = extrema(spacing_per_dim)
    if min_spacing ≈ max_spacing
        return kgrid_from_minimal_spacing(lattice, min_spacing)
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
        kgrid_larger = kgrid_from_minimal_spacing(lattice, spacing + 1e-4)
        if prod(kgrid_larger) ≥ n_kpoints
            return kgrid_larger
        else
            return kgrid_from_minimal_spacing(lattice, spacing)
        end
    end
end
function kgrid_from_minimal_kpoints(model::Model, args...)
    kgrid_from_minimal_kpoints(model.lattice, args...)
end
