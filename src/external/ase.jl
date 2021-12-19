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
            lattice[i, j] = austrip(cell_julia[j][i] * u"Å")
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
    map(ase_atoms_translation_map(pyobj)) do (symbol, atom_indices)
        coords = Vec3{T}[]
        for idx in atom_indices
            push!(coords, frac_coords[idx, :])
        end
        ElementCoulomb(symbol) => coords
    end
end


function load_magnetic_moments_ase(pyobj::PyObject)
    @assert pyisinstance(pyobj, pyimport("ase").Atoms)
    magmoms = pyobj.get_initial_magnetic_moments()
    map(ase_atoms_translation_map(pyobj)) do (symbol, atom_indices)
        ElementCoulomb(symbol) => [magmoms[i] for i in atom_indices]
    end
end


"""
Map translating between the `atoms` datastructure used in DFTK
and the indices of the respective atoms in the ASE.Atoms object
in `pyobj`.
"""
function ase_atoms_translation_map(pyobj::PyObject)
    @assert pyisinstance(pyobj, pyimport("ase").Atoms)
    numbers = pyobj.get_atomic_numbers()
    map(unique(numbers)) do atnum
        ElementCoulomb(atnum).symbol => findall(isequal(atnum), numbers)
    end
end


#                                                             convert A.U. -> Å
ase_cell(lattice) = pyimport("ase").cell.Cell(Array(lattice)' / austrip(1u"Å"))
ase_cell(model::Model) = ase_cell(model.lattice)


function ase_atoms(lattice_or_model, atoms, magnetic_moments=nothing)
    cell = ase_cell(lattice_or_model)
    symbols = String[]
    for (elem, pos) in atoms
        append!(symbols, fill(string(atomic_symbol(elem)), length(pos)))
    end
    scaled_positions = vcat([pos for (elem, pos) in atoms]...)

    magmoms = nothing
    if !isnothing(magnetic_moments)
        @assert length(magnetic_moments) == length(atoms)
        for (elem, magmom) in magnetic_moments
            @assert all(m -> m isa Number, magmom)
        end
        magmoms = vcat([magmom for (elem, magmom) in magnetic_moments]...)
        @assert length(magmoms) == length(scaled_positions)
    end

    pyimport("ase").Atoms(symbols=symbols, cell=cell, pbc=true,
                          scaled_positions=hcat(scaled_positions...)',
                          magmoms=magmoms)
end
function ase_atoms(model::Model, magnetic_moments=nothing)
    ase_atoms(model.lattice, model.atoms, magnetic_moments)
end
