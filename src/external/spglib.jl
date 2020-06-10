using PyCall
# Routines for interaction with spglib

"""
Construct a tuple containing the lattice and the positions of the species
in the convention required to take the place of a `cell` datastructure used in spglib.
"""
function spglib_cell_atommapping_(lattice, atoms)
    lattice = Matrix{Float64}(lattice)  # spglib operates in double precision
    n_attypes = isempty(atoms) ? 0 : sum(length(positions) for (type, positions) in atoms)
    spg_numbers = Vector{Int}(undef, n_attypes)
    spg_positions = Matrix{Float64}(undef, n_attypes, 3)

    offset = 0
    nextnumber = 1
    atommapping = Dict{Int, Any}()
    for (iatom, (type, positions)) in enumerate(atoms)
        atommapping[nextnumber] = type
        for (ipos, pos) in enumerate(positions)
            # assign the same number to all types with this position
            spg_numbers[offset + ipos] = nextnumber
            spg_positions[offset + ipos, :] = pos
        end
        offset += length(positions)
        nextnumber += 1
    end

    # Note: In the python interface of spglib the lattice vectors
    #       are given in rows, but DFTK uses columns
    #       For future reference: The C interface spglib also uses columns.
    (lattice', spg_positions, spg_numbers), atommapping
end
spglib_cell(lattice, atoms) = first(spglib_cell_atommapping_(lattice, atoms))


@timing "spglib_get_symmetry" function spglib_get_symmetry(lattice, atoms; tol_symmetry=1e-5)
    spglib = pyimport("spglib")
    lattice = Matrix{Float64}(lattice)  # spglib operates in double precision

    if isempty(atoms)
        # spglib doesn't like no atoms, so we default to
        # no symmetries (even though there are lots)
        return [Mat3{Int}(I)], [Vec3(zeros(3))]
    end

    # Ask spglib for symmetry operations and for irreducible mesh
    spg_symops = spglib.get_symmetry(spglib_cell(lattice, atoms),
                                     symprec=tol_symmetry)

    # If spglib does not find symmetries give an error
    if spg_symops === nothing
        err_message=spglib.get_error_message()
        error("spglib failed to get the symmetries. Check your lattice, use a " *
              "uniform BZ mesh or disable symmetries. Spglib reported : " * err_message)
    end

    Stildes = [St for St in eachslice(spg_symops["rotations"]; dims=1)]
    τtildes = [rationalize.(τt, tol=tol_symmetry)
               for τt in eachslice(spg_symops["translations"]; dims=1)]
    @assert length(τtildes) == length(Stildes)

    # Checks: (A Stilde A^{-1}) is unitary
    for Stilde in Stildes
        Scart = lattice * Stilde * inv(lattice)  # Form S in cartesian coords
        if maximum(abs, Scart'Scart - I) > tol_symmetry
            error("spglib returned non-unitary rotation matrix")
        end
    end

    # Check (Stilde, τtilde) maps atoms to equivalent atoms in the lattice
    for (Stilde, τtilde) in zip(Stildes, τtildes)
        for (elem, positions) in atoms
            for coord in positions
                diffs = [rationalize.(Stilde * coord + τtilde - pos, tol=tol_symmetry)
                         for pos in positions]

                # If all elements of a difference in diffs is integer, then
                # Stilde * coord + τtilde and pos are equivalent lattice positions
                if !any(all(isinteger, d) for d in diffs)
                    error("Cannot map the atom at position $coord to another atom of the " *
                          "same element under the symmetry operation (Stilde, τtilde):\n" *
                          "($Stilde, $τtilde)")
                end
            end
        end
    end

    Stildes, τtildes
end

function spglib_standardize_cell(lattice::MatT, atoms; correct_symmetry=true,
                                 primitive=false, tol_symmetry=1e-5) where {MatT}
    spglib = pyimport("spglib")
    T = eltype(lattice)

    # Convert lattice and atoms to spglib and keep the mapping between our atoms
    # and spglibs atoms
    cell, atommapping = spglib_cell_atommapping_(lattice, atoms)

    # Ask spglib to standardize the cell (i.e. find a cell, which fits the spglib conventions)
    res = spglib.standardize_cell(spglib_cell(lattice, atoms), to_primitive=primitive,
                                  no_idealize=!correct_symmetry, symprec=tol_symmetry)
    spg_lattice, spg_scaled_positions, spg_numbers = res

    # Note: In the python interface of spglib the lattice vectors
    #       are given in rows, but DFTK uses columns
    #       For future reference: The C interface spglib also uses columns.
    newlattice = MatT(spg_lattice')
    newatoms = [(atommapping[iatom]
                 => T.(spg_scaled_positions[findall(isequal(iatom), spg_numbers), :]))
                for iatom in unique(spg_numbers)]
    newlattice, newatoms
end
