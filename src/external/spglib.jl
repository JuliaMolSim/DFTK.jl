# Routines for interaction with spglib
# Note: spglib/C uses the row-major convention, thus we need to perform transposes
#       between julia and spglib (https://spglib.github.io/spglib/variable.html)
const SPGLIB = spglib_jll.libsymspg

function spglib_get_error_message()
    error_code = ccall((:spg_get_error_code, SPGLIB), Cint, ())
    return unsafe_string(ccall((:spg_get_error_message, SPGLIB), Cstring, (Cint,), error_code))
end

"""
Construct a tuple containing the positions of the species
in the convention required to take the place of a `cell` datastructure used in spglib.
"""
function spglib_atommapping(atoms)
    n_attypes = isempty(atoms) ? 0 : sum(length(positions) for (typ, positions) in atoms)
    spg_numbers = Vector{Cint}(undef, n_attypes)
    spg_positions = Matrix{Cdouble}(undef, 3, n_attypes)

    offset = 0
    nextnumber = 1
    atommapping = Dict{Int, Any}()
    for (iatom, (type, positions)) in enumerate(atoms)
        atommapping[nextnumber] = type
        for (ipos, pos) in enumerate(positions)
            # assign the same number to all types with this position
            spg_numbers[offset + ipos] = nextnumber
            spg_positions[:, offset + ipos] .= pos
        end
        offset += length(positions)
        nextnumber += 1
    end

    (spg_positions, spg_numbers), atommapping
end
spglib_atoms(atoms) = first(spglib_atommapping(atoms))


@timing function spglib_get_symmetry(lattice, atoms; tol_symmetry=1e-5)
    lattice = Matrix{Float64}(lattice)  # spglib operates in double precision

    if isempty(atoms)
        # spglib doesn't like no atoms, so we default to
        # no symmetries (even though there are lots)
        return [Mat3{Int}(I)], [Vec3(zeros(3))]
    end

    # Ask spglib for symmetry operations and for irreducible mesh
    spg_positions, spg_numbers = spglib_atoms(atoms)

    spg_n_ops = ccall((:spg_get_multiplicity, SPGLIB), Cint,
        (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, Cint, Cdouble),
        copy(lattice'), spg_positions, spg_numbers, Cint(length(spg_numbers)), tol_symmetry)

    spg_rotations    = Array{Cint}(undef, 3, 3, spg_n_ops)
    spg_translations = Array{Cdouble}(undef, 3, spg_n_ops)

    ccall((:spg_get_symmetry, SPGLIB), Cint,
         (Ptr{Cint}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, Cint, Cdouble),
         spg_rotations, spg_translations, spg_n_ops, copy(lattice'), spg_positions, spg_numbers,
         Cint(length(spg_numbers)), tol_symmetry)

    # If spglib does not find symmetries give an error
    if spg_n_ops == 0
        err_message = spglib_get_error_message()
        error("spglib failed to get the symmetries. Check your lattice, use a " *
              "uniform BZ mesh or disable symmetries. Spglib reported : " * err_message)
    end

    # Note: Transposes are performed to convert between spglib row-major to julia column-major
    Stildes = [Mat3{Int}(spg_rotations[:, :, i]') for i in 1:spg_n_ops]
    τtildes = [rationalize.(Vec3{Float64}(spg_translations[:, i]), tol=tol_symmetry)
               for i in 1:spg_n_ops]

    # Checks: (A Stilde A^{-1}) is unitary
    for Stilde in Stildes
        Scart = lattice * Stilde * inv(lattice)  # Form S in cartesian coords
        if maximum(abs, Scart'Scart - I) > tol_symmetry
            error("spglib returned bad symmetries: Non-unitary rotation matrix.")
        end
    end

    # Check (Stilde, τtilde) maps atoms to equivalent atoms in the lattice
    for (Stilde, τtilde) in zip(Stildes, τtildes)
        for (elem, positions) in atoms
            for coord in positions
                diffs = [rationalize.(Stilde * coord + τtilde - pos, tol=5*tol_symmetry)
                         for pos in positions]

                # If all elements of a difference in diffs is integer, then
                # Stilde * coord + τtilde and pos are equivalent lattice positions
                if !any(all(isinteger, d) for d in diffs)
                    error("spglib returned bad symmetries: Cannot map the atom at position " *
                          "$coord to another atom of the same element under the symmetry " *
                          "operation (Stilde, τtilde):\n" *
                          "($Stilde, $τtilde)")
                end
            end
        end
    end

    Stildes, τtildes
end

function spglib_standardize_cell(lattice::AbstractArray{T}, atoms; correct_symmetry=true,
                                 primitive=false, tol_symmetry=1e-5) where {T}
    # Convert lattice and atoms to spglib and keep the mapping between our atoms
    spg_lattice = copy(Matrix{Float64}(lattice)')
    # and spglibs atoms
    spg_atoms, atommapping = spglib_atommapping(atoms)
    spg_positions, spg_numbers = spg_atoms

    # Ask spglib to standardize the cell (i.e. find a cell, which fits the spglib conventions)
    num_atoms = ccall((:spg_standardize_cell, SPGLIB), Cint,
      (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, Cint, Cint, Cint, Cdouble),
      spg_lattice, spg_positions, spg_numbers, length(spg_numbers), Cint(primitive),
      Cint(!correct_symmetry), tol_symmetry)
    spg_lattice = copy(spg_lattice')

    newatoms = [(atommapping[iatom]
                 => T.(spg_positions[findall(isequal(iatom), spg_numbers), :]))
                for iatom in unique(spg_numbers)]
    Matrix{T}(spg_lattice), newatoms
end

function spglib_get_stabilized_reciprocal_mesh(kgrid_size, rotations::Vector;
                                               is_shift=Vec3(0, 0, 0),
                                               is_time_reversal=false,
                                               qpoints=[Vec3(0.0, 0.0, 0.0)],
                                               isdense=false)
    spg_rotations = cat([copy(Cint.(S')) for S in rotations]..., dims=3)
    nkpt = prod(kgrid_size)
    mapping = Vector{Cint}(undef, nkpt)
    grid_address = Matrix{Cint}(undef, 3, nkpt)

    nrot = length(rotations)
    n_kpts = ccall((:spg_get_stabilized_reciprocal_mesh, SPGLIB), Cint,
      (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Cint, Cint, Ptr{Cint}, Cint, Ptr{Cdouble}),
      grid_address, mapping, [Cint.(kgrid_size)...], [Cint.(is_shift)...], Cint(is_time_reversal),
      Cint(nrot), spg_rotations, Cint(length(qpoints)), Vec3{Float64}.(qpoints))

    return n_kpts, Int.(mapping), [Vec3{Int}(grid_address[:, i]) for i in 1:nkpt]
end
