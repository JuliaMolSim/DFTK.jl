# Routines for interaction with pymatgen, e.g. converting to
# its structures from the DFTK equivalents

function pymatgen_lattice(lattice::AbstractArray)
    # Notice: Pymatgen uses rows as lattice vectors, so we unpeel
    # our lattice column by column. The default unit in pymatgen is Ångström
    Lattice = pyimport("pymatgen.core.lattice").Lattice
    bohr_to_A = 1 / austrip(1u"Å")
    Lattice([Array(bohr_to_A .* lattice[:, i]) for i in 1:3])
end
pymatgen_lattice(model::Model) = pymatgen_lattice(model.lattice)


function pymatgen_structure(model_or_lattice, atoms)
    pylattice = pymatgen_lattice(model_or_lattice)

    n_species = sum(length(pos) for (spec, pos) in atoms)
    pyspecies = Vector{Int}(undef, n_species)
    pypositions = Array{Vector{Float64}}(undef, n_species)
    ispec = 1
    for (spec, pos) in atoms
        for coord in pos
            pyspecies[ispec] = charge_nuclear(spec)
            pypositions[ispec] = Vector{Float64}(coord)
            ispec = ispec + 1
        end
    end
    @assert ispec == n_species + 1

    Structure = pyimport("pymatgen.core.structure").Structure
    Structure(pylattice, pyspecies, pypositions)
end
pymatgen_structure(model::Model) = pymatgen_structure(model, model.atoms)


function load_lattice_pymatgen(T, pyobj::PyObject)
    Structure = pyimport("pymatgen.core.structure").Structure
    Lattice   = pyimport("pymatgen.core.lattice").Lattice

    if pyisinstance(pyobj, Structure)
        load_lattice_pymatgen(T, pyobj.lattice)
    elseif pyisinstance(pyobj, Lattice)
        lattice = Matrix{T}(undef, 3, 3)
        for i in 1:3, j in 1:3
            lattice[i, j] = austrip(get(get(pyobj.matrix, j-1), i-1) * u"Å")
        end
        Mat3{T}(lattice)
    else
        error("load_lattice_pymatgen not implemented for python type $pyobj")
    end
end


"""
Load a DFTK-compatible atoms representation from a supported pymatgen object.
All atoms are using a Coulomb model.
"""
function load_atoms_pymatgen(T, pyobj::PyObject)
    @assert pyisinstance(pyobj, pyimport("pymatgen.core.structure").Structure)
    map(unique(pyobj.species)) do spec
        coords = [s.frac_coords for s in pyobj.sites if s.specie == spec]
        ElementCoulomb(spec.number) => coords
    end
end
