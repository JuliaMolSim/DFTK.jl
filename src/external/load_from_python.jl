"""
Load a DFTK-compatible lattice object from a supported python object (e.g. pymatgen or ASE)
"""
function load_lattice(T, pyobj::PyObject)
    mg = pyimport_e("pymatgen")
    ase = pyimport_e("ase")

    if !ispynull(mg)
        mg_supported = (mg.Structure, mg.Lattice)
        if any(pyisinstance(pyobj, s) for s in mg_supported)
            return load_lattice_pymatgen(T, pyobj)
        end
    end

    if !ispynull(ase)
        ase_supported = (ase.Atoms, ase.cell.Cell)
        if any(pyisinstance(pyobj, s) for s in ase_supported)
            return load_lattice_ase(T, pyobj)
        end
    end

    error("load_lattice not implemented for python type $pyobj")
end


function load_atoms(T, pyobj::PyObject)
    mg = pyimport_e("pymatgen")
    ase = pyimport_e("ase")

    if !ispynull(mg) && pyisinstance(pyobj, mg.Structure)
        return load_atoms_pymatgen(T, pyobj)
    end
    if !ispynull(ase) && pyisinstance(pyobj, ase.Atoms)
        return load_atoms_ase(T, pyobj)
    end

    error("load_atoms not implemented for python type $pyobj")
end
