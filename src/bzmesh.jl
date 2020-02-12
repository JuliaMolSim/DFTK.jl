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

    Stildes = spg_symops["rotations"]
    τtildes = spg_symops["translations"]
    n_symops = size(Stildes, 1)
    @assert size(τtildes, 1) == n_symops
    # Notice: In the language of the latex document in the docs
    # spglib returns \tilde{S} and \tilde{τ} in integer real-space coordinates, such that
    # (A Stilde A^{-1}) is the actual \tilde{S} from the document as a unitary matrix.
    #
    # Still we have the following properties for S and τ given is *integer* and
    # *fractional* real-space coordinates:
    #      - \tilde{S}^{-1} = S^T (if applied to a vector in frac. coords in reciprocal space)

    # Checks: (A Stilde A^{-1}) is unitary
    for Stilde in eachslice(Stildes; dims=1)
        Scart = lattice * Stilde * inv(lattice)  # Form S in cartesian coords
        if maximum(abs, Scart'Scart - I) > sqrt(eps(Float64))
            error("spglib returned non-unitary rotation matrix")
        end
    end

    # Check (Stilde, τtilde) maps atoms to equivalent atoms in the lattice
    for (Stilde, τtilde) in zip(eachslice(Stildes; dims=1), eachslice(τtildes; dims=1))
        for (elem, positions) in atoms
            # Rationalize atomic positions
            positions = [rationalize.(pos, tol=tol_symmetry) for pos in positions]
            for coord in positions
                mapped = rationalize.(Stilde * coord + τtilde, tol=tol_symmetry)

                # mapped is on the lattice if it maps to one other lattice position
                # of the same element
                if !any(all(isinteger, mapped - pos) for pos in positions)
                    error("Cannot map the atom at position $coord to another atom of the " *
                          "same element under the symmetry operation (Stilde, τtilde):\n" *
                          "($Stilde, $τtilde)")
                end
            end
        end
    end

    mapping, grid = spglib.get_stabilized_reciprocal_mesh(
        kgrid_size, Stildes, is_shift=[0, 0, 0], is_time_reversal=true
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

            isym = findfirst(1:n_symops) do idx
                # If the difference between kred and Stilde' * k == Stilde^{-1} * k
                # is only integer in fractional reciprocal-space coordinates, then
                # kred and S' * k are equivalent k-Points
                all(isinteger, kred - (Stildes[idx, :, :]' * k))
            end

            if isym === nothing
                error("No corresponding symop found for $k -> $kred. This is a bug.")
            else
                S = Stildes[isym, :, :]'  # in fractional reciprocal coordinates
                # τ in fractional real-space coordinates:
                τ = -inv(Stildes[isym, :, :]) * τtildes[isym, :, :]
                push!(ksymops[ik], (S, τ))
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
