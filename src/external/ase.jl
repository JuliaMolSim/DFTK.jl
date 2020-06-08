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
            lattice[i, j] = units.Ǎ * cell_julia[j][i]
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


ase_cell(lattice) = pyimport("ase").cell.Cell(Array(lattice)' / DFTK.units.Å)
ase_cell(model::Model) = ase_cell(model.lattice)


function ase_atoms(lattice_or_model, atoms)
    cell = ase_cell(lattice_or_model)
    symbols = String[]
    for (elem, pos) in atoms
        append!(symbols, fill(string(elem.symbol), length(pos)))
    end
    scaled_positions = vcat([pos for (elem, pos) in atoms]...)
    pyimport("ase").Atoms(symbols=symbols, cell=cell, pbc=true,
                          scaled_positions=hcat(scaled_positions...)')
end
ase_atoms(model::Model) = pymatgen_structure(model.lattice, model.atoms)
