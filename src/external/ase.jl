#                                                             convert A.U. -> Å
ase_cell(lattice) = pyimport("ase").cell.Cell(Array(lattice)' / austrip(1u"Å"))
ase_cell(model::Model) = ase_cell(model.lattice)

function ase_atoms(lattice_or_model, atoms, positions, magnetic_moments=[])
    if isempty(magnetic_moments)  # Collect Z magnetic moments
        magmoms = nothing
    else
        magmoms = [normalize_magnetic_moment(mom)[3] for mom in magnetic_moments]
    end
    cell    = ase_cell(lattice_or_model)
    symbols = string.(atomic_symbol.(atoms))
    scaled_positions = reduce(hcat, positions)'
    # Hack using `info` to be able to determine the elements with `load_atoms`.
    # Two elements with same atomic symbol should be equal.
    element_mapping = Dict()
    for atom in atoms
        symbol = atomic_symbol(atom)
        @assert !isnothing(symbol)
        symbol ∈ keys(element_mapping) && @assert element_mapping[symbol] == atom
        element_mapping[symbol] = atom
    end
    pyimport("ase").Atoms(;symbols, cell, pbc=true, scaled_positions, magmoms,
                          info=element_mapping)
end
function ase_atoms(model::Model, magnetic_moments=[])
    ase_atoms(model.lattice, model.atoms, model.positions, magnetic_moments)
end


function load_lattice_ase(pyobj::PyObject)
    ase = pyimport("ase")
    if pyisinstance(pyobj, ase.Atoms)
        if !all(pyobj.pbc)
            error("DFTK only supports calculations with periodic boundary conditions.")
        end
        load_lattice_ase(pyobj.cell)
    elseif pyisinstance(pyobj, ase.cell.Cell)
        lattice = zeros(3, 3)
        cell_julia = convert(Array, pyobj)  # Array of arrays
        for i = 1:3, j = 1:3
            lattice[i, j] = austrip(cell_julia[j][i] * u"Å")
        end
        Mat3(lattice)
    else
        error("load_lattice_ase not implemented for python type $pyobj")
    end
end


function load_atoms_ase(pyobj::PyObject)
    @assert pyisinstance(pyobj, pyimport("ase").Atoms)
    # TODO Be smarter and look at the calculator to determine the psps
    # We try to give the correct `Element` types by looking at the `info` field.
    # If it is not possible, we give back `ElementCoulomb` by default.
    info = pyobj.info
    if typeof(info) <: Dict && !isempty(info)
        @assert all(isa.(keys(info), String))
        @assert all(isa.(values(info), Element))
        [info[symbol] for symbol in pyobj.get_chemical_symbols()]
    else
        [ElementCoulomb(number) for number in pyobj.get_atomic_numbers()]
    end
end


function load_positions_ase(pyobj::PyObject)
    @assert pyisinstance(pyobj, pyimport("ase").Atoms)
    [Vec3(pos) for pos in eachrow(pyobj.get_scaled_positions())]
end


function load_magnetic_moments_ase(pyobj::PyObject)
    @assert pyisinstance(pyobj, pyimport("ase").Atoms)
    [normalize_magnetic_moment(magmom)
     for magmom in pyobj.get_initial_magnetic_moments()]
end
