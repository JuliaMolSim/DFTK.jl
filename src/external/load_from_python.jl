"""
Load a DFTK-compatible lattice object from a supported python object (e.g. pymatgen or ASE)
"""
function load_lattice(T, pyobj::PyObject)
    mg = pyimport_e("pymatgen")
    ase = pyimport_e("ase")

    if !ispynull(mg)
        Lattice   = pyimport("pymatgen.core.lattice").Lattice
        Structure = pyimport("pymatgen.core.structure").Structure
        if any(pyisinstance(pyobj, s) for s in (Lattice, Structure))
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

    if !ispynull(mg)
        if pyisinstance(pyobj, pyimport("pymatgen.core.structure").Structure)
            return load_atoms_pymatgen(T, pyobj)
        end
    end
    if !ispynull(ase) && pyisinstance(pyobj, ase.Atoms)
        return load_atoms_ase(T, pyobj)
    end

    error("load_atoms not implemented for python type $pyobj")
end


function load_magnetic_moments(pyobj::PyObject)
    ase = pyimport_e("ase")
    if !ispynull(ase) && pyisinstance(pyobj, ase.Atoms)
        return load_magnetic_moments_ase(pyobj)
    end
    error("load_magnetic_moments not implemented for python type $pyobj")
end
