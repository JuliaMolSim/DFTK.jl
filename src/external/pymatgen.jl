function pymatgen_lattice(lattice::AbstractArray)
    # Notice: Pymatgen uses rows as lattice vectors, so we unpeel
    # our lattice column by column. The default unit in pymatgen is Ångström
    Lattice = pyimport("pymatgen.core.lattice").Lattice
    bohr_to_A = 1 / austrip(1u"Å")
    Lattice([Array(bohr_to_A .* lattice[:, i]) for i in 1:3])
end
pymatgen_lattice(model::Model) = pymatgen_lattice(model.lattice)


function pymatgen_structure(model_or_lattice, atoms, positions)
    Structure = pyimport("pymatgen.core.structure").Structure
    Structure(pymatgen_lattice(model_or_lattice),
              charge_nuclear.(atoms),
              Vector{Float64}.(positions))
end
pymatgen_structure(model::Model) = pymatgen_structure(model, model.atoms, model.positions)


function load_lattice_pymatgen(pyobj::PyObject)
    Structure = pyimport("pymatgen.core.structure").Structure
    Lattice   = pyimport("pymatgen.core.lattice").Lattice

    if pyisinstance(pyobj, Structure)
        load_lattice_pymatgen(pyobj.lattice)
    elseif pyisinstance(pyobj, Lattice)
        lattice = zeros(3, 3)
        for i in 1:3, j in 1:3
            lattice[i, j] = austrip(get(get(pyobj.matrix, j-1), i-1) * u"Å")
        end
        Mat3(lattice)
    else
        error("load_lattice_pymatgen not implemented for python type $pyobj")
    end
end


function load_atoms_pymatgen(pyobj::PyObject)
    @assert pyisinstance(pyobj, pyimport("pymatgen.core.structure").Structure)
    [ElementCoulomb(spec.number) for spec in pyobj.species]
end


function load_positions_pymatgen(pyobj::PyObject)
    @assert pyisinstance(pyobj, pyimport("pymatgen.core.structure").Structure)
    [Vec3{Float64}(site.frac_coords) for site in pyobj.sites]
end
