using PyCall
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


@doc raw"""
    bzmesh_ir_wedge(kgrid_size, lattice, atoms; tol_symmetry=1e-5)

Construct the irreducible wedge of a uniform Brillouin zone mesh for sampling ``k``-Points.
The function returns a tuple `(kcoords, ksymops)`, where `kcoords` are the list of
irreducible ``k``-Points and `ksymops` are a list of symmetry operations for regenerating
the full mesh. `lattice` are the lattice vectors, column by column, `atoms` are pairs
representing a mapping from `Element` objects to a list of positions in fractional
coordinates. `tol_symmetry` is the tolerance used for searching for symmetry operations.
"""
function bzmesh_ir_wedge(kgrid_size, lattice, atoms; tol_symmetry=1e-5)
    all(isequal.(kgrid_size, 1)) && return bzmesh_uniform(kgrid_size)
    spglib = pyimport("spglib")

    # Ask spglib for symmetry operations and for irreducible mesh
    spg_symops = spglib.get_symmetry(spglib_cell(lattice, atoms),
                                     symprec=tol_symmetry)

    # If spglib does not find symmetries give an error
    spg_symops !== nothing || error(
        "spglib failed to get the symmetries. Check your lattice, or use a uniform BZ mesh."
    )

    # Checks: S is unitary (noting that S is given in lattice coordinates)
    for S in eachslice(spg_symops["rotations"]; dims=1)
        Scart = lattice * S * inv(lattice)  # Form S in cartesian coords
        if maximum(abs, Scart'Scart - I) > sqrt(eps(Float64))
            error("spglib returned non-unitary rotation matrix")
        end
    end
    # TODO check that S * lattice + τ ∈ lattice

    mapping, grid = spglib.get_stabilized_reciprocal_mesh(
        kgrid_size, spg_symops["rotations"], is_shift=[0, 0, 0], is_time_reversal=true
    )

    # Convert irreducible k-Points to DFTK conventions
    kgrid_size = Vec3{Int}(kgrid_size)
    kcoords = [Vec3{Int}(grid[ik + 1, :]) .// kgrid_size for ik in unique(mapping)]

    # Find the indices of the corresponding reducible k-Points in `grid`, which one of the
    # irreducible k-Points in `kcoords` generates.
    k_all_reducible = [findall(isequal(elem), mapping) for elem in unique(mapping)]

    # ksymops will be the list of symmetry operations (for each irreducible k-Point)
    # needed to do the respective mapping irreducible -> reducible to get the respective
    # entry in `k_all_reducible`.
    SymOp = Tuple{Mat3{Int}, Vec3{Float64}}
    ksymops = Vector{Vector{SymOp}}(undef, length(kcoords))
    for (ik, k) in enumerate(kcoords)
        ksymops[ik] = Vector{SymOp}()
        for ired in k_all_reducible[ik]
            kred = Vec3(grid[ired, :]) .// kgrid_size

            isym = findfirst(1:size(spg_symops["rotations"], 1)) do idx
                # If the denominator of the difference between
                # kred and S' * k is 1, than the difference is an integer,
                # i.e. kred and S' * k are equivalent k-Points
                S = view(spg_symops["rotations"], idx, :, :)
                all(elem.den == 1 for elem in (kred - (S' * k)))
            end

            if isym === nothing
                error("No corresponding symop found for $k -> $kred. This is a bug.")
            else
                S = spg_symops["rotations"][isym, :, :]
                τ = spg_symops["translations"][isym, :]
                push!(ksymops[ik], (S', -inv(S) * τ))
            end
        end
    end

    # The symmetry operation (S == I and τ == 0) should be present for each k-Point
    @assert all(nothing !== findfirst(Sτ -> iszero(Sτ[1] - I) && iszero(Sτ[2]), ops)
                for ops in ksymops)

    kcoords, ksymops
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
