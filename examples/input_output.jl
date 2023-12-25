# # Input and output formats

# This section provides an overview of the input and output formats
# supported by DFTK, usually via integration with a third-party library.
#
# ## Reading / writing files supported by AtomsIO
# [AtomsIO](https://github.com/mfherbst/AtomsIO.jl) is a Julia package which supports
# reading / writing atomistic structures from / to a large range of file formats.
# Supported formats include Crystallographic Information Framework (CIF),
# XYZ and extxyz files, ASE / Gromacs / LAMMPS / Amber trajectory files
# or input files of various other codes (e.g. Quantum Espresso, VASP, ABINIT, CASTEP, …).
# The full list of formats is is available in the
# [AtomsIO documentation](https://mfherbst.github.io/AtomsIO.jl/stable).
#
# The AtomsIO functionality is split into two packages. The main package, `AtomsIO` itself,
# only depends on packages, which are registered in the Julia General package registry.
# In contrast `AtomsIOPython` extends `AtomsIO` by parsers depending on python packages,
# which are automatically managed via `PythonCall`. While it thus provides the full set of
# supported IO formats, this also adds additional practical complications, so some users
# may choose not to use `AtomsIOPython`.
#
# As an example we start the calculation of a simple antiferromagnetic iron crystal
# using a Quantum-Espresso input file, [Fe_afm.pwi](Fe_afm.pwi).
# For more details about calculations on magnetic systems
# using collinear spin, see [Collinear spin and magnetic systems](@ref).
#
# First we parse the Quantum Espresso input file using AtomsIO,
# which reads the lattice, atomic positions and initial magnetisation
# from the input file and returns it as an
# [AtomsBase](https://github.com/JuliaMolSim/AtomsBase.jl) `AbstractSystem`,
# the JuliaMolSim community standard for representing atomic systems.

using AtomsIO        # Use Julia-only IO parsers
using AtomsIOPython  # Use python-based IO parsers (e.g. ASE)
system = load_system("Fe_afm.pwi")

# Next we attach pseudopotential information, since currently the parser is not
# yet capable to read this information from the file.

using DFTK
system = attach_psp(system, Fe="hgh/pbe/fe-q16.hgh")

# Finally we make use of DFTK's [AtomsBase integration](@ref) to run the calculation.

model = model_LDA(system; temperature=0.01)
basis = PlaneWaveBasis(model; Ecut=10, kgrid=(2, 2, 2))
ρ0 = guess_density(basis, system)
scfres = self_consistent_field(basis, ρ=ρ0);

# !!! warning "DFTK data formats are not yet fully matured"
#     The data format in which DFTK saves data as well as the general interface
#     of the [`load_scfres`](@ref) and [`save_scfres`](@ref) pair of functions
#     are not yet fully matured. If you use the functions or the produced files
#     expect that you need to adapt your routines in the future even with patch
#     version bumps.

# ## Writing VTK files for visualization
# For visualizing the density or the Kohn-Sham orbitals DFTK supports storing
# the result of an SCF calculations in the form of VTK files.
# These can afterwards be visualized using tools such
# as [paraview](https://www.paraview.org/).
# Using this feature requires
# the [WriteVTK.jl](https://github.com/jipolanco/WriteVTK.jl/) Julia package.

using WriteVTK
save_scfres("iron_afm.vts", scfres; save_ψ=true);

# This will save the iron calculation above into the file `iron_afm.vts`,
# using `save_ψ=true` to also include the KS orbitals.

# ## Parsable data-export using json
# Many structures in DFTK support the (unexported) `todict` function,
# which returns a simplified dictionary representation of the data.

DFTK.todict(scfres.energies)

# This in turn can be easily written to disk using a JSON library.
# Currently we integrate most closely with `JSON3`,
# which is thus recommended.

using JSON3
open("iron_afm_energies.json", "w") do io
    JSON3.pretty(io, DFTK.todict(scfres.energies))
end
println(read("iron_afm_energies.json", String))

# Once JSON3 is loaded, additionally a convenience function for saving
# a summary of `scfres` objects using `save_scfres` is available:

using JSON3
save_scfres("iron_afm.json", scfres)

# Similarly a summary of the band data (occupations, eigenvalues, εF, etc.)
# for post-processing can be dumped using `save_bands`:
save_bands("iron_afm_scfres.json", scfres)

# Notably this function works both for the results obtained
# by `self_consistent_field` as well as `compute_bands`:
bands = compute_bands(scfres, kline_density=10)
save_bands("iron_afm_bands.json", bands)

# ## Writing and reading JLD2 files
# The full state of a DFTK self-consistent field calculation can be
# stored on disk in form of an [JLD2.jl](https://github.com/JuliaIO/JLD2.jl) file.
# This file can be read from other Julia scripts
# as well as other external codes supporting the HDF5 file format
# (since the JLD2 format is based on HDF5). This includes notably `h5py`
# to read DFTK output from python.

using JLD2
save_scfres("iron_afm.jld2", scfres)

# Saving such JLD2 files supports some options, such as `save_ψ=false`, which avoids saving
# the Bloch waves (much faster and smaller files). Notice that JLD2 files can also be used
# with [`save_bands`](@ref).

# Since such JLD2 can also be read by DFTK to start or continue a calculation,
# these can also be used for checkpointing or for transferring results
# to a different computer.
# See [Saving SCF results on disk and SCF checkpoints](@ref) for details.

# (Cleanup files generated by this notebook.)
rm.(["iron_afm.vts", "iron_afm.jld2",
     "iron_afm.json", "iron_afm_energies.json", "iron_afm_scfres.json",
     "iron_afm_bands.json"])
