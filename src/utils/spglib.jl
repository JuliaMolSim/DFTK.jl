# Routines for interaction with spglib

"""
Construct a tuple containing the lattice and the positions of the species
in the convention required to take the place of a `cell` datastructure used in spglib.
"""
function spglib_cell(lattice, composition...)
    n_species = sum(length(positions) for (spec, positions) in composition)
    spg_numbers = Vector{Int}(undef, n_species)
    spg_positions = Matrix{Float64}(undef, n_species, 3)

    offset = 0
    nextnumber = 1
    for (spec, positions) in composition
        for (i, pos) in enumerate(positions)
            # assign the same number to all species with this position
            spg_numbers[offset + i] = nextnumber
            spg_positions[offset + i, :] = pos
        end
        offset += length(positions)
        nextnumber += 1
    end

    # Note: In the python interface of spglib the lattice vectors
    #       are given in rows, but DFTK uses columns
    #       For future reference: The C interface spglib also uses columns.
    #
    # But also: python/C are row-major and Julia column-major, so no transpose here.
    (lattice, spg_positions, spg_numbers)
end
