# Routines for interaction with spglib

"""
Construct a tuple containing the lattice and the positions of the species
in the convention required to take the place of a `cell` datastructure used in spglib.
"""
function spglib_cell(lattice, atoms)
    n_attypes = sum(length(positions) for (type, positions) in atoms)
    spg_numbers = Vector{Int}(undef, n_attypes)
    spg_positions = Matrix{Float64}(undef, n_attypes, 3)

    offset = 0
    nextnumber = 1
    for (type, positions) in atoms
        for (i, pos) in enumerate(positions)
            # assign the same number to all types with this position
            spg_numbers[offset + i] = nextnumber
            spg_positions[offset + i, :] = pos
        end
        offset += length(positions)
        nextnumber += 1
    end

    # Note: In the python interface of spglib the lattice vectors
    #       are given in rows, but DFTK uses columns
    #       For future reference: The C interface spglib also uses columns.
    (lattice', spg_positions, spg_numbers)
end
