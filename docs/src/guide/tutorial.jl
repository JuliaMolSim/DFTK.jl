# # Tutorial

#nb # DFTK is a Julia package for playing with plane-wave
#nb # density-functional theory algorithms. In its basic formulation it
#nb # solves periodic Kohn-Sham equations.
#
# This document provides an overview of the structure of the code
# and how to access basic information about calculations.
# Basic familiarity with the concepts of plane-wave density functional theory
# is assumed throughout. Feel free to take a look at the
#md # [Periodic problems](@ref periodic-problems)
#nb # [Periodic problems](https://docs.dftk.org/stable/guide/periodic_problems/)
# or the
#md # [Introductory resources](@ref introductory-resources)
#nb # [Introductory resources](https://docs.dftk.org/stable/guide/introductory_resources/)
# chapters for some introductory material on the topic.
#
# !!! note "Convergence parameters in the documentation"
#     We use rough parameters in order to be able
#     to automatically generate this documentation very quickly.
#     Therefore results are far from converged.
#     Tighter thresholds and larger grids should be used for more realistic results.
#
# For our discussion we will use the classic example of computing the LDA ground state
# of the [silicon crystal](https://www.materialsproject.org/materials/mp-149).
# Performing such a calculation roughly proceeds in three steps.

using DFTK
using Plots
using Unitful
using UnitfulAtomic
using PseudoPotentialData

## 1. Define lattice and atomic positions
a = 5.431u"angstrom"          # Silicon lattice constant
lattice = a / 2 * [[0 1 1.];  # Silicon lattice vectors
                   [1 0 1.];  # specified column by column
                   [1 1 0.]];

# By default, all numbers passed as arguments are assumed to be in atomic
# units.  Quantities such as temperature, energy cutoffs, lattice vectors, and
# the k-point grid spacing can optionally be annotated with Unitful units,
# which are automatically converted to the atomic units used internally. For
# more details, see the [Unitful package
# documentation](https://painterqubits.github.io/Unitful.jl/stable/) and the
# [UnitfulAtomic.jl package](https://github.com/sostock/UnitfulAtomic.jl).

# We use a pseudodojo pseudopotential
# (see [PseudoPotentialData](https://github.com/JuliaMolSim/PseudoPotentialData.jl)
#  for more details on `PseudoFamily`):

pd_lda_family = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
Si = ElementPsp(:Si, pd_lda_family)

## Specify type and positions of atoms
atoms     = [Si, Si]
positions = [ones(3)/8, -ones(3)/8]

# Note that DFTK supports a few other ways to supply atomistic structures,
# see for example the sections on [AtomsBase integration](@ref)
# and [Input and output formats](@ref) for details.

## 2. Select model and basis
model = model_DFT(lattice, atoms, positions; functionals=LDA())
kgrid = [4, 4, 4]     # k-point grid (Regular Monkhorst-Pack grid)
Ecut = 7              # kinetic energy cutoff
## Ecut = 190.5u"eV"  # Could also use eV or other energy-compatible units
basis = PlaneWaveBasis(model; Ecut, kgrid)
## Note the implicit passing of keyword arguments here:
## this is equivalent to PlaneWaveBasis(model; Ecut=Ecut, kgrid=kgrid)

## 3. Run the SCF procedure to obtain the ground state
scfres = self_consistent_field(basis, tol=1e-5);

# That's it! Now you can get various quantities from the result of the SCF.
# For instance, the different components of the energy:
scfres.energies

# Eigenvalues:
stack(scfres.eigenvalues)
# `eigenvalues` is an array (indexed by k-points) of arrays (indexed by
# eigenvalue number).
#
# The resulting matrix is 7 (number of computed eigenvalues) by 8
# (number of irreducible k-points). There are 7 eigenvalues per
# k-point because there are 4 occupied states in the system (4 valence
# electrons per silicon atom, two atoms per unit cell, and paired
# spins), and the eigensolver gives itself some breathing room by
# computing some extra states (see the `bands` argument to
# `self_consistent_field` as well as the [`AdaptiveBands`](@ref) documentation).
# There are only 8 k-points (instead of 4x4x4) because symmetry has been used to reduce the
# amount of computations to just the irreducible k-points (see
#md # [Crystal symmetries](@ref)
#nb # [Crystal symmetries](https://docs.dftk.org/stable/developer/symmetries/)
# for details).
#
# We can check the occupations ...
stack(scfres.occupation)
# ... and density, where we use that the density objects in DFTK are
# indexed as ρ[ix, iy, iz, iσ], i.e. first in the 3-dimensional real-space grid
# and then in the spin component.
rvecs = collect(r_vectors(basis))[:, 1, 1]  # slice along the x axis
x = [r[1] for r in rvecs]                   # only keep the x coordinate
plot(x, scfres.ρ[:, 1, 1, 1], label="", xlabel="x", ylabel="ρ", marker=2)

# We can also perform various postprocessing steps:
# We can get the Cartesian forces (in Hartree / Bohr):
compute_forces_cart(scfres)
# As expected, they are numerically zero in this highly symmetric configuration.
# We could also compute a band structure,
plot_bandstructure(scfres; kline_density=10)
# or plot a density of states, for which we increase the kgrid a bit
# to get smoother plots:
bands = compute_bands(scfres, MonkhorstPack(6, 6, 6))
plot_dos(bands; temperature=1e-3, smearing=Smearing.FermiDirac())
# Note that directly employing the `scfres` also works, but the results
# are much cruder:
plot_dos(scfres; temperature=1e-3, smearing=Smearing.FermiDirac())

# !!! info "Where to go from here"
#     - **Background on DFT:**
#       * [Periodic problems](@ref periodic-problems),
#       * [Introduction to density-functional theory](@ref),
#       * [Self-consistent field methods](@ref)
#     - **Running calculations:**
#       * [Temperature and metallic systems](@ref metallic-systems)
#       * [Pseudopotentials](@ref)
#       * [Performing a convergence study](@ref)
#       * [Geometry optimization](@ref)
#       * [AtomsBase integration](@ref) and wider ecosystem: Building / reading structures etc.
#     - **Tips and tricks:**
#       * [Using DFTK on compute clusters](@ref),
#       * [Using DFTK on GPUs](@ref),
#       * [Saving SCF results on disk and SCF checkpoints](@ref)
