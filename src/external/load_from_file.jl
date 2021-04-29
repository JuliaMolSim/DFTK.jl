#
# Load DFTK-compatible structural information from an external file
# Relies on ASE and other external libraries to do the parsing
#
using PyCall


function load_lattice(T, file::AbstractString)
    ase = pyimport_e("ase")
    ispynull(ase) && error("Install ASE to load data from exteral files")
    load_lattice(T, pyimport("ase.io").read(file))
end

function load_atoms(T, file::AbstractString)
    ase = pyimport_e("ase")
    ispynull(ase) && error("Install ASE to load data from exteral files")
    load_atoms(T, pyimport("ase.io").read(file))
end

function load_magnetic_moments(file::AbstractString)
    ase = pyimport_e("ase")
    ispynull(ase) && error("Install ASE to load data from exteral files")
    load_magnetic_moments(pyimport("ase.io").read(file))
end

"""
Return a DFTK-compatible `lattice` object loaded from an ASE `Atoms`, a pymatgen `Structure`
or a compatible file (e.g. xyz file, cif file etc.)
"""
load_lattice(args...; kwargs...) = load_lattice(Float64, args...; kwargs...)

"""
Return a DFTK-compatible `atoms` object loaded from an ASE `Atoms`, a pymatgen `Structure`
or a compatible file (e.g. xyz file, cif file etc.)
"""
load_atoms(  args...; kwargs...) = load_atoms(  Float64, args...; kwargs...)

load_model(  args...; kwargs...) = load_model(  Float64, args...; kwargs...)
load_basis(  args...; kwargs...) = load_basis(  Float64, args...; kwargs...)
load_density(args...; kwargs...) = load_density(Float64, args...; kwargs...)
