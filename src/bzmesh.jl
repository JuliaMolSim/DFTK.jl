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
    kcoords, [[(Mat3{Int}(I), Vec3(zeros(3)))] for _ in 1:length(kcoords)]
end


const SymOp = Tuple{Mat3{Int}, Vec3{Float64}}
"""
Implements a primitive search to find an irreducible subset of kpoints
amongst the provided kpoints.
"""
function find_irreducible_kpoints(kcoords, Stildes, τtildes)
    #
    # This function is required, because spglib sometimes flags kpoints
    # as reducible, where we cannot find a symmetry operation to generate
    # them from the provided irreducible kpoints.
    #

    # Flag which kpoints have already been mapped to another irred.
    # kpoint or which have been decided to be irreducible.
    kcoords_mapped = zeros(Bool, length(kcoords))
    kirreds = empty(kcoords)           # Container for irreducible kpoints
    ksymops = Vector{Vector{SymOp}}()  # Corresponding symops

    while !all(kcoords_mapped)
        # Select next not mapped kpoint as irreducible
        ik = findfirst(isequal(false), kcoords_mapped)
        push!(kirreds, kcoords[ik])
        thisk_symops = [(Mat3{Int}(I), Vec3(zeros(3)))]
        kcoords_mapped[ik] = true

        for jk in findall(.!kcoords_mapped)
            isym = findfirst(1:length(Stildes)) do isym
                # If the difference between kred and Stilde' * k == Stilde^{-1} * k
                # is only integer in fractional reciprocal-space coordinates, then
                # kred and S' * k are equivalent k-Points
                all(isinteger, kcoords[jk] - (Stildes[isym]' * kcoords[ik]))
            end

            if !isnothing(isym)  # Found a reducible kpoint
                kcoords_mapped[jk] = true
                S = Stildes[isym]'                  # in fractional reciprocal coordinates
                τ = -Stildes[isym] \ τtildes[isym]  # in fractional real-space coordinates
                τ = τ .- floor.(τ)
                @assert all(0 .≤ τ .< 1)
                push!(thisk_symops, (S, τ))
            end
        end  # jk

        push!(ksymops, thisk_symops)
    end
    kirreds, ksymops
end

@doc raw"""
Return the ``k``-point symmetry operations associated to a lattice, model or basis.
Since the ``k``-point discretisations may break some of the symmetries, the latter
case will return a subset of the symmetries of the former two.
"""
function symmetry_operations(lattice, atoms; tol_symmetry=1e-5, kcoords=nothing)
    symops = []
    Stildes, τtildes = spglib_get_symmetry(lattice, atoms, tol_symmetry=tol_symmetry)

    # Notice: In the language of the latex document in the docs
    # spglib returns \tilde{S} and \tilde{τ} in integer real-space coordinates, such that
    # (A Stilde A^{-1}) is the actual \tilde{S} from the document as a unitary matrix.
    #
    # Still we have the following properties for S and τ given in *integer* and
    # *fractional* real-space coordinates:
    #      - \tilde{S}^{-1} = S^T (if applied to a vector in frac. coords in reciprocal space)

    for isym = 1:length(Stildes)
        S = Stildes[isym]'                  # in fractional reciprocal coordinates
        τ = -Stildes[isym] \ τtildes[isym]  # in fractional real-space coordinates
        τ = τ .- floor.(τ)
        @assert all(0 .≤ τ .< 1)
        push!(symops, (S, τ))
    end

    symops = unique(symops)

    
    if kcoords !== nothing
        # filter only the operations that respect the symmetries of the discrete BZ grid
        function preserves_grid(S)
            all(normalize_kpoint_coordinate(S * k) in kcoords
                for k in normalize_kpoint_coordinate.(kcoords))
        end
        symops = filter(symop -> preserves_grid(symop[1]), symops)
    end


    symops
end
symmetry_operations(model::Model; kwargs...) = symmetry_operations(model.lattice, model.atoms; kwargs...)


@doc raw"""
    bzmesh_ir_wedge(kgrid_size, lattice, atoms; tol_symmetry=1e-5)

Construct the irreducible wedge of a uniform Brillouin zone mesh for sampling ``k``-Points.
The function returns a tuple `(kcoords, ksymops)`, where `kcoords` are the list of
irreducible ``k``-Points and `ksymops` are a list of symmetry operations for regenerating
the full mesh. `lattice` are the lattice vectors, column by column, `atoms` are pairs
representing a mapping from `Element` objects to a list of positions in fractional
coordinates. `tol_symmetry` is the tolerance used for searching for symmetry operations.
"""
function bzmesh_ir_wedge(kgrid_size, lattice, atoms;
                         tol_symmetry=1e-5, kshift=[0, 0, 0])
    all(isequal.(kgrid_size, 1)) && return bzmesh_uniform(kgrid_size, kshift=kshift)

    # Transform kshift to the convention used in spglib:
    #    If is_shift is set (i.e. integer 1), then a shift of 0.5 is performed,
    #    else no shift is performed along an axis.
    kshift = Vec3{Rational{Int}}(kshift)
    all(ks in (0, 1//2) for ks in kshift) || error("Only kshifts of 0 or 1//2 implemented.")

    kpoints_mp = kgrid_monkhorst_pack(kgrid_size, kshift=kshift)

    # Get the list of symmetry operations (S,τ) that preserve the MP grid
    symops = symmetry_operations(lattice, atoms; kcoords=kpoints_mp)

    # Give the remaining symmetries to spglib to compute an irreducible k-Point mesh
    # TODO implement time-reversal symmetry and turn the flag to true
    is_shift = Int.(2 * kshift)
    Stildes = [s[1]' for s in symops]
    spg_rotations = permutedims(cat(Stildes..., dims=3), (3, 1, 2))
    mapping, grid = pyimport("spglib").get_stabilized_reciprocal_mesh(
        kgrid_size, spg_rotations, is_shift=is_shift, is_time_reversal=false
    )

    # Convert irreducible k-Points to DFTK conventions
    kgrid_size = Vec3{Int}(kgrid_size)
    kirreds = [(kshift .+ Vec3{Int}(grid[ik + 1, :])) .// kgrid_size
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
            kred = (kshift .+ Vec3(grid[ired, :])) .// kgrid_size

            # Note that this relies on the identity coming up first in symops
            isym = findfirst(symops) do symop
                # If the difference between kred and Stilde' * k == Stilde^{-1} * k
                # is only integer in fractional reciprocal-space coordinates, then
                # kred and S' * k are equivalent k-Points
                all(isinteger, kred - (symop[1] * k))
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
        Stildes = [s[1]' for s in symops]
        τtildes = [-s[1]' * s[2] for s in symops]
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

    kirreds, ksymops
end


@doc raw"""
Apply various standardisations to a lattice and a list of atoms. It uses spglib to detect
symmetries (within `tol_symmetry`), then cleans up the lattice according to the symmetries
(unless `correct_symmetry` is `false`) and returns the resulting standard lattice
and atoms. If `primitive` is `true` (default) the primitive unit cell is returned, else
the conventional unit cell is returned.
"""
function standardize_atoms(lattice, atoms; correct_symmetry=true, primitive=true,
                           tol_symmetry=1e-5)
    spglib_standardize_cell(lattice, atoms; correct_symmetry=correct_symmetry,
                            primitive=primitive, tol_symmetry=tol_symmetry)
end


@doc raw"""
Selects a kgrid_size to ensure a minimal spacing (in inverse Bohrs) between kpoints.
Default is ``2π * 0.04 \AA^{-1}``.
"""
function kgrid_size_from_minimal_spacing(lattice, spacing=2π * 0.04 / units.Ǎ)
    for d in 1:3
        @assert norm(lattice[:, d]) != 0
        # Otherwise the formula for the reciprocal lattice
        # computation is not correct
    end
    recip_lattice = 2π * inv(lattice')
    [ceil(Int, norm(recip_lattice[:, i]) ./ spacing) for i = 1:3]
end
