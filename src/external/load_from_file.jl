#
# Load DFTK-compatible structural information from an external file
# Relies on ASE and other external libraries to do the parsing
#
using PyCall


function load_lattice(file::AbstractString)
    ase = pyimport_e("ase")
    ispynull(ase) && error("Install ASE to load data from exteral files")
    load_lattice(pyimport("ase.io").read(file))
end


function load_atoms(file::AbstractString)
    ase = pyimport_e("ase")
    ispynull(ase) && error("Install ASE to load data from exteral files")
    load_atoms(pyimport("ase.io").read(file))
end

function load_magnetic_moments(file::AbstractString)
    ase = pyimport_e("ase")
    ispynull(ase) && error("Install ASE to load data from exteral files")
    load_magnetic_moments(pyimport("ase.io").read(file))
end
