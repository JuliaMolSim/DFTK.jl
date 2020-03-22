function load_lattice_ase(T, pyobj::PyObject)
    ase = pyimport("ase")

    if pyisinstance(pyobj, ase.Atoms)
        if !all(pyobj.pbc)
            error("DFTK only supports calculations with periodic boundary conditions.")
        end
        load_lattice_ase(T, pyobj.cell)
    elseif pyisinstance(pyobj, ase.cell.Cell)
        lattice = zeros(3, 3)
        cell_julia = convert(Array, pyobj)  # Array of arrays
        for i = 1:3, j = 1:3
            lattice[i, j] = units.Ǎ * cell_julia[j][i]  # TODO check ordering!
        end
        Mat3{T}(lattice)
    else
        error("load_lattice_ase not implemented for python type $pyobj")
    end
end


function load_atoms_ase(T, pyobj::PyObject)
    @assert pyisinstance(pyobj, pyimport("ase").Atoms)
    # TODO Be smarter and look at the calculator to determine the psps

    frac_coords = pyobj.get_scaled_positions()
    numbers = pyobj.get_atomic_numbers()
    map(unique(numbers)) do atnum
        coords = Vec3{T}[]
        for idx in findall(isequal(atnum), numbers)
            push!(coords, frac_coords[idx, :])
        end
        ElementCoulomb(atnum) => coords
    end
end
