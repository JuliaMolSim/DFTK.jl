using PyCall
# Routines for interaction with spglib

function import_spglib()
    spglib = pyimport("spglib")
    version = VersionNumber(spglib.__version__)

    if version < v"1.12"
        @warn "Spglib below 1.12 not tested with DFTK" maxlog=1
    elseif v"1.14" <= version < v"1.15"
        @warn "Spglib $version is known to be faulty when used with DFTK." maxlog=1
    end
    spglib
end

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

function spglib_get_symmetry(lattice, atoms; tol_symmetry=1e-5)
    spglib = import_spglib()
    lattice = Matrix{Float64}(lattice)  # spglib operates in double precision

    # Ask spglib for symmetry operations and for irreducible mesh
    spg_symops = spglib.get_symmetry(spglib_cell(lattice, atoms),
                                     symprec=tol_symmetry)

    # If spglib does not find symmetries give an error
    spg_symops !== nothing || error(
        "spglib failed to get the symmetries. Check your lattice, or use a uniform BZ mesh."
    )

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
