using AtomsIO
using ASEconvert
export load_lattice
export load_atoms
export load_positions
export load_magnetic_moments

load_lattice(file::AbstractString)   = load_lattice(ase.io.read(file))
load_atoms(file::AbstractString)     = load_atoms(ase.io.read(file))
load_positions(file::AbstractString) = load_positions(ase.io.read(file))
load_magnetic_moments(file::AbstractString) = load_magnetic_moments(ase.io.read(file))

function load_deprecated(symbol)
    Base.depwarn("$symbol is deprecated. Use AtomsIO.load_system instead. " *
                 "See https://docs.dftk.org/stable/examples/atomsbase/ for more details.",
                 symbol)
end

function load_lattice(pyobj::ASEconvert.Py)
    load_deprecated(:load_lattice)
    parse_system(pyconvert(AbstractSystem, pyobj)).lattice
end
function load_atoms(pyobj::ASEconvert.Py)
    load_deprecated(:load_atoms)
    parse_system(pyconvert(AbstractSystem, pyobj)).atoms
end
function load_positions(pyobj::ASEconvert.Py)
    load_deprecated(:load_positions)
    parse_system(pyconvert(AbstractSystem, pyobj)).positions
end
function load_magnetic_moments(pyobj::ASEconvert.Py)
    load_deprecated(:load_magnetic_moments)
    parse_system(pyconvert(AbstractSystem, pyobj)).magnetic_moments
end

@deprecate(ase_atoms(lattice, atoms, positions, magnetic_moments=[]),
           convert_ase(atomic_system(lattice, atoms, positions, magnetic_moments)))
@deprecate(ase_atoms(model::Model, magnetic_moments=[]),
           convert_ase(atomic_system(model, magnetic_moments)))
