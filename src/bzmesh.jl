include("external/spglib.jl")

@doc raw"""
    bzmesh_uniform(kgrid_size)

Construct a uniform Brillouin zone mesh for sampling the ``k``-Points. The function
returns a tuple `(kcoords, ksymops)`, where `kcoords` are the list of ``k``-Points
and `ksymops` are a list of symmetry operations (for interface compatibility
with `PlaneWaveBasis` and `bzmesh_irreducible`. No symmetry reduction is attempted,
such that there will be `prod(kgrid_size)` ``k``-Points returned and all symmetry
operations are the identity.
"""
function bzmesh_uniform(kgrid_size)
    kgrid_size = Vec3{Int}(kgrid_size)
    start = -floor.(Int, (kgrid_size .- 1) ./ 2)
    stop  = ceil.(Int, (kgrid_size .- 1) ./ 2)
    kcoords = [Vec3([i, j, k] .// kgrid_size) for i=start[1]:stop[1],
               j=start[2]:stop[2], k=start[3]:stop[3]]
    ksymops = [[(Mat3{Int}(I), Vec3(zeros(3)))] for _ in 1:length(kcoords)]
    vec(kcoords), ksymops
end


const SymOp = Tuple{Mat3{Int}, Vec3{Float64}}
"""
Implements a primitive search to find an irreducible subset of kpoints
amongst the provided kpoints.
"""
function find_irreducible_kpoints(kpoints, Stildes, τtildes)
    #
    # This function is required, because spglib sometimes flags kpoints
    # as reducible, where we cannot find a symmetry operation to generate
    # them from the provided irreducible kpoints.
    #
    n_symops = length(Stildes)

    # Flag which kpoints have already been mapped to another irred.
    # kpoint or which have been decided to be irreducible.
    kpoints_mapped = zeros(Bool, length(kpoints))
    kirreds = empty(kpoints)           # Container for irreducible kpoints
    ksymops = Vector{Vector{SymOp}}()  # Corresponding symops

    while !all(kpoints_mapped)
        # Select next not mapped kpoint as irreducible
        ik = findfirst(isequal(false), kpoints_mapped)
        push!(kirreds, kpoints[ik])
        thisk_symops = [(Mat3{Int}(I), Vec3(zeros(3)))]
        kpoints_mapped[ik] = true

        for jk in findall(.!kpoints_mapped)
            isym = findfirst(1:n_symops) do isym
                # If the difference between kred and Stilde' * k == Stilde^{-1} * k
                # is only integer in fractional reciprocal-space coordinates, then
                # kred and S' * k are equivalent k-Points
                all(isinteger, kpoints[jk] - (Stildes[isym]' * kpoints[ik]))
            end

            if !isnothing(isym)  # Found a reducible kpoint
                kpoints_mapped[jk] = true
                S = Stildes[isym]'                  # in fractional reciprocal coordinates
                τ = -Stildes[isym] \ τtildes[isym]  # in fractional real-space coordinates
                push!(thisk_symops, (S, τ))
            end
        end  # jk

        push!(ksymops, thisk_symops)
    end
    kirreds, ksymops
end



@doc raw"""
    bzmesh_ir_wedge(kgrid_size, lattice, atoms; tol_symmetry=1e-5)

Construct the irreducible wedge of a uniform Brillouin zone mesh for sampling ``k``-Points.
The function returns a tuple `(kcoords, ksymops)`, where `kcoords` are the list of
irreducible ``k``-Points and `ksymops` are a list of symmetry operations for regenerating
the full mesh. `lattice` are the lattice vectors, column by column, `atoms` are pairs
representing a mapping from `AbstractElement` objects to a list of positions in fractional
coordinates. `tol_symmetry` is the tolerance used for searching for symmetry operations.
"""
function bzmesh_ir_wedge(kgrid_size, lattice, atoms; tol_symmetry=1e-5)
    all(isequal.(kgrid_size, 1)) && return bzmesh_uniform(kgrid_size)

    # Notice: In the language of the latex document in the docs
    # spglib returns \tilde{S} and \tilde{τ} in integer real-space coordinates, such that
    # (A Stilde A^{-1}) is the actual \tilde{S} from the document as a unitary matrix.
    #
    # Still we have the following properties for S and τ given in *integer* and
    # *fractional* real-space coordinates:
    #      - \tilde{S}^{-1} = S^T (if applied to a vector in frac. coords in reciprocal space)

    Stildes, τtildes = spglib_get_symmetry(lattice, atoms; tol_symmetry=tol_symmetry)
    n_symops = length(Stildes)

    # Use determined symmetries to deduce irreducible k-Point mesh
    spg_rotations = permutedims(cat(Stildes..., dims=3), (3, 1, 2))
    mapping, grid = import_spglib().get_stabilized_reciprocal_mesh(
        kgrid_size, spg_rotations, is_shift=[0, 0, 0], is_time_reversal=true
    )

    # Convert irreducible k-Points to DFTK conventions
    kgrid_size = Vec3{Int}(kgrid_size)
    kirreds = [Vec3{Int}(grid[ik + 1, :]) .// kgrid_size for ik in unique(mapping)]

    # Find the indices of the corresponding reducible k-Points in `grid`, which one of the
    # irreducible k-Points in `kirreds` generates.
    k_all_reducible = [findall(isequal(elem), mapping) for elem in unique(mapping)]

    # This list collects a list of extra reducible k-Points, which could not be
    # mapped to any irreducible kpoint yet even though spglib claims this can be done.
    kreds_notmapped = empty(kirreds)

    # ksymops will be the list of symmetry operations (for each irreducible k-Point)
    # needed to do the respective mapping irreducible -> reducible to get the respective
    # entry in `k_all_reducible`.
    ksymops = Vector{Vector{SymOp}}(undef, length(kirreds))
    for (ik, k) in enumerate(kirreds)
        ksymops[ik] = Vector{SymOp}()
        for ired in k_all_reducible[ik]
            kred = Vec3(grid[ired, :]) .// kgrid_size

            isym = findfirst(1:n_symops) do isym
                # If the difference between kred and Stilde' * k == Stilde^{-1} * k
                # is only integer in fractional reciprocal-space coordinates, then
                # kred and S' * k are equivalent k-Points
                all(isinteger, kred - (Stildes[isym]' * k))
            end

            if isym === nothing  # No symop found for $k -> $kred
                push!(kreds_notmapped, kred)
            else
                S = Stildes[isym]'                  # in fractional reciprocal coordinates
                τ = -Stildes[isym] \ τtildes[isym]  # in fractional real-space coordinates
                push!(ksymops[ik], (S, τ))
            end
        end
    end

    if !isempty(kreds_notmapped)
        eirreds, esymops = find_irreducible_kpoints(kreds_notmapped, Stildes, τtildes)

        @info("$(length(kreds_notmapped)) reducible kpoints could not be generated from " *
              "the irreducible kpoints returned by spglib. $(length(eirreds)) of " *
              "these are added as extra irreducible kpoints.")

        append!(kirreds, eirreds)
        append!(ksymops, esymops)
    end

    # The symmetry operation (S == I and τ == 0) should be present for each k-Point
    @assert all(nothing !== findfirst(Sτ -> iszero(Sτ[1] - I) && iszero(Sτ[2]), ops)
                for ops in ksymops)

    kirreds, ksymops
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
