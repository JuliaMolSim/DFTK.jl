#
# Load DFTK-compatible structural information from an external file
# Relies on ASE and other external libraries to do the parsing
#
using PyCall

function load_from_file(payload::Function, file::AbstractString)
    ase = pyimport_e("ase")
    ispynull(ase) && error("Install ASE to load data from exteral files")
    payload(pyimport("ase.io").read(file))
end
load_lattice(file::AbstractString)   = load_from_file(load_lattice,   file)
load_atoms(file::AbstractString)     = load_from_file(load_atoms,     file)
load_positions(file::AbstractString) = load_from_file(load_positions, file)
load_magnetic_moments(file::AbstractString) = load_from_file(load_magnetic_moments, file)


function load_lattice(pyobj::PyObject)
    mg = pyimport_e("pymatgen")
    ase = pyimport_e("ase")

    if !ispynull(mg)
        Lattice   = pyimport("pymatgen.core.lattice").Lattice
        Structure = pyimport("pymatgen.core.structure").Structure
        if any(pyisinstance(pyobj, s) for s in (Lattice, Structure))
            return load_lattice_pymatgen(pyobj)
        end
    end

    if !ispynull(ase)
        ase_supported = (ase.Atoms, ase.cell.Cell)
        if any(pyisinstance(pyobj, s) for s in ase_supported)
            return load_lattice_ase(pyobj)
        end
    end

    error("load_lattice not implemented for python type $pyobj")
end

function load_atoms(pyobj::PyObject)
    mg = pyimport_e("pymatgen")
    ase = pyimport_e("ase")

    if !ispynull(mg)
        if pyisinstance(pyobj, pyimport("pymatgen.core.structure").Structure)
            return load_atoms_pymatgen(pyobj)
        end
    end
    if !ispynull(ase) && pyisinstance(pyobj, ase.Atoms)
        return load_atoms_ase(pyobj)
    end

    error("load_atoms not implemented for python type $pyobj")
end

function load_positions(pyobj::PyObject)
    mg = pyimport_e("pymatgen")
    ase = pyimport_e("ase")

    if !ispynull(mg)
        if pyisinstance(pyobj, pyimport("pymatgen.core.structure").Structure)
            return load_positions_pymatgen(pyobj)
        end
    end
    if !ispynull(ase) && pyisinstance(pyobj, ase.Atoms)
        return load_positions_ase(pyobj)
    end

    error("load_positions not implemented for python type $pyobj")
end


function load_magnetic_moments(pyobj::PyObject)
    ase = pyimport_e("ase")
    if !ispynull(ase) && pyisinstance(pyobj, ase.Atoms)
        return load_magnetic_moments_ase(pyobj)
    end
    error("load_magnetic_moments not implemented for python type $pyobj")
end
