include("external/spglib.jl")


"""Bring kpoint coordinates into the range [-0.5, 0.5)"""
function normalize_kpoint_coordinate(x::Real)
    x = x - round(Int, x, RoundNearestTiesUp)
    @assert -0.5 ≤ x < 0.5
    x
end
normalize_kpoint_coordinate(k::AbstractVector) = normalize_kpoint_coordinate.(k)


"""Construct the coordinates of the kpoints in a (shifted) Monkorst-Pack grid"""
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
    bzmesh_uniform(kgrid_size)

Construct a uniform Brillouin zone mesh for sampling the ``k``-Points. The function
returns a tuple `(kcoords, ksymops)`, where `kcoords` are the list of ``k``-Points
and `ksymops` are a list of symmetry operations (for interface compatibility
with `PlaneWaveBasis` and `bzmesh_irreducible`. No symmetry reduction is attempted,
such that there will be `prod(kgrid_size)` ``k``-Points returned and all symmetry
operations are the identity.
"""
function bzmesh_uniform(kgrid_size; kshift=[0, 0, 0])
    kcoords = kgrid_monkhorst_pack(kgrid_size; kshift=kshift)
    kcoords, [[identity_symop()] for _ in 1:length(kcoords)], [identity_symop()]
end


@doc raw"""
    bzmesh_ir_wedge(kgrid_size, lattice, atoms; tol_symmetry=1e-5)

Construct the irreducible wedge of a uniform Brillouin zone mesh for sampling ``k``-Points.
The function returns a tuple `(kcoords, ksymops)`, where `kcoords` are the list of
irreducible ``k``-Points and `ksymops` are a list of symmetry operations for regenerating
the full mesh. `lattice` are the lattice vectors, column by column, `atoms` are pairs
representing a mapping from `Element` objects to a list of positions in fractional
coordinates. `tol_symmetry` is the tolerance used for searching for symmetry operations.
"""
function bzmesh_ir_wedge(kgrid_size, symops; kshift=[0, 0, 0])
    all(isequal.(kgrid_size, 1)) && return bzmesh_uniform(kgrid_size, kshift=kshift)

    # Transform kshift to the convention used in spglib:
    #    If is_shift is set (i.e. integer 1), then a shift of 0.5 is performed,
    #    else no shift is performed along an axis.
    kshift = Vec3{Rational{Int}}(kshift)
    all(ks in (0, 1//2) for ks in kshift) || error("Only kshifts of 0 or 1//2 implemented.")

    kpoints_mp = kgrid_monkhorst_pack(kgrid_size, kshift=kshift)

    # Filter those symmetry operations (S,τ) that preserve the MP grid
    symops = symops_preserving_kgrid(symops, kpoints_mp)

    # Give the remaining symmetries to spglib to compute an irreducible k-Point mesh
    # TODO implement time-reversal symmetry and turn the flag to true
    is_shift = Int.(2 * kshift)
    Stildes = [S for (S, τ) in symops]
    spg_rotations = cat(Stildes..., dims=3)
    _, mapping, grid = spglib_get_stabilized_reciprocal_mesh(
        kgrid_size, spg_rotations, is_shift=is_shift, is_time_reversal=false
    )
    # Convert irreducible k-Points to DFTK conventions
    kgrid_size = Vec3{Int}(kgrid_size)
    kirreds = [(kshift .+ grid[ik + 1]) .// kgrid_size
               for ik in unique(mapping)]
    kirreds = normalize_kpoint_coordinate.(kirreds)

    # Find the indices of the corresponding reducible k-Points in `grid`, which one of the
    # irreducible k-Points in `kirreds` generates.
    k_all_reducible = [findall(isequal(elem), mapping) for elem in unique(mapping)]

    # This list collects a list of extra reducible k-Points, which could not be
    # mapped to any irreducible kpoint yet even though spglib claims this can be done.
    # This happens because spglib actually fails for non-ideal cases, resulting
    # in *wrong results* being returned. See the discussion in
    # https://github.com/spglib/spglib/issues/101
    kreds_notmapped = empty(kirreds)

    # ksymops will be the list of symmetry operations (for each irreducible k-Point)
    # needed to do the respective mapping irreducible -> reducible to get the respective
    # entry in `k_all_reducible`.
    ksymops = Vector{Vector{SymOp}}(undef, length(kirreds))
    for (ik, k) in enumerate(kirreds)
        ksymops[ik] = Vector{SymOp}()
        for ired in k_all_reducible[ik]
            kred = (kshift .+ grid[ired]) .// kgrid_size

            # Note that this relies on the identity coming up first in symops
            isym = findfirst(symops) do symop
                # If the difference between kred and Stilde' * k == Stilde^{-1} * k
                # is only integer in fractional reciprocal-space coordinates, then
                # kred and S' * k are equivalent k-Points
                S = symop[1]
                all(isinteger, kred - (S * k))
            end

            if isym === nothing  # No symop found for $k -> $kred
                push!(kreds_notmapped, normalize_kpoint_coordinate(kred))
            else
                push!(ksymops[ik], symops[isym])
            end
        end
    end

    if !isempty(kreds_notmapped)
        # add them as reducible anyway
        Stildes = [S' for (S, τ) in symops]
        τtildes = [-S' * τ for (S, τ) in symops]
        eirreds, esymops = find_irreducible_kpoints(kreds_notmapped, Stildes, τtildes)
        @info("$(length(kreds_notmapped)) reducible kpoints could not be generated from " *
              "the irreducible kpoints returned by spglib. $(length(eirreds)) of " *
              "these are added as extra irreducible kpoints.")

        append!(kirreds, eirreds)
        append!(ksymops, esymops)
    end

    # The symmetry operation (S == I and τ == 0) should be present for each k-Point
    @assert all(findfirst(Sτ -> iszero(Sτ[1] - I) && iszero(Sτ[2]), ops) !== nothing
                for ops in ksymops)

    kirreds, ksymops, symops
end


@doc raw"""
Apply various standardisations to a lattice and a list of atoms. It uses spglib to detect
symmetries (within `tol_symmetry`), then cleans up the lattice according to the symmetries
(unless `correct_symmetry` is `false`) and returns the resulting standard lattice
and atoms. If `primitive` is `true` (default) the primitive unit cell is returned, else
the conventional unit cell is returned.
"""
const standardize_atoms = spglib_standardize_cell

@doc raw"""
Selects a kgrid_size to ensure a minimal spacing (in inverse Bohrs) between kpoints.
Default is ``2π * 0.04 \AA^{-1}``.
"""
function kgrid_size_from_minimal_spacing(lattice, spacing=2π * 0.04 / units.Ǎ)
    @assert spacing > 0
    isinf(spacing) && return [1, 1, 1]

    for d in 1:3
        @assert norm(lattice[:, d]) != 0
        # Otherwise the formula for the reciprocal lattice
        # computation is not correct
    end
    recip_lattice = 2π * inv(lattice')
    [ceil(Int, norm(recip_lattice[:, i]) ./ spacing) for i = 1:3]
end
