# # Tutorial
#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/guide/@__NAME__.ipynb)
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/guide/@__NAME__.ipynb)

#nb # DFTK is a Julia package for playing with plane-wave
#nb # density-functional theory algorithms. In its basic formulation it
#nb # solves periodic Kohn-Sham equations.
#
# This document provides an overview of the structure of the code
# and how to access basic information about calculations.
# Basic familiarity with the concepts of plane-wave density functional theory
# is assumed throughout. Feel free to take a look at the
#md # [Periodic problems](@ref periodic-problems)
#nb # [Periodic problems](https://juliamolsim.github.io/DFTK.jl/stable/guide/periodic_problems/)
# or the
#md # [density-functional theory](@ref density-functional-theory)
#nb # [density-functional theory](https://juliamolsim.github.io/DFTK.jl/stable/guide/density_functional_theory/)
# chapters for some introductory material on the topic.
#
# !!! note "Convergence parameters in the documentation"
#     We use rough parameters in order to be able
#     to automatically generate this documentation very quickly.
#     Therefore results are far from converged.
#     Tighter thresholds and larger grids should be used for more realistic results.
#
# For our discussion we will use the classic example of
# computing the LDA ground state of the
# [silicon crystal](https://www.materialsproject.org/materials/mp-149).
# Performing such a calculation roughly proceeds in three steps.

using DFTK
using Plots
using Unitful
using UnitfulAtomic

## 1. Define lattice and atomic positions
a = 5.431u"angstrom"          # Silicon lattice constant
lattice = a / 2 * [[0 1 1.];  # Silicon lattice vectors
                   [1 0 1.];  # specified column by column
                   [1 1 0.]]

# By default, all numbers passed as arguments are assumed to be in atomic
# units.  Quantities such as temperature, energy cutoffs, lattice vectors, and
# the k-point grid spacing can optionally be annotated with Unitful units,
# which are automatically converted to the atomic units used internally. For
# more details, see the [Unitful package
# documentation](https://painterqubits.github.io/Unitful.jl/stable/) and the
# [UnitfulAtomic.jl package](https://github.com/sostock/UnitfulAtomic.jl).

## Load HGH pseudopotential for Silicon
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))

## Specify type and positions of atoms
atoms = [Si => [ones(3)/8, -ones(3)/8]]

## 2. Select model and basis
model = model_LDA(lattice, atoms)
kgrid = [4, 4, 4]     # k-point grid (Regular Monkhorst-Pack grid)
Ecut = 7              # kinetic energy cutoff
## Ecut = 190.5u"eV"  # Could also use eV or other energy-compatible units
basis = PlaneWaveBasis(model; Ecut, kgrid)
## Note the implicit passing of keyword arguments here:
## this is equivalent to PlaneWaveBasis(model; Ecut=Ecut, kgrid=kgrid)

## 3. Run the SCF procedure to obtain the ground state
scfres = self_consistent_field(basis, tol=1e-8);

# That's it! Now you can get various quantities from the result of the SCF.
# For instance, the different components of the energy:
scfres.energies

# Eigenvalues: 
hcat(scfres.eigenvalues...)
# `eigenvalues` is an array (indexed by k-points) of arrays (indexed by
# eigenvalue number). The "splatting" operation `...` calls `hcat`
# with all the inner arrays as arguments, which collects them into a
# matrix.
#
# The resulting matrix is 7 (number of computed eigenvalues) by 8
# (number of k-points). There are 7 eigenvalues per k-point because
# there are 4 occupied states in the system (4 valence electrons per
# silicon atom, two atoms per unit cell, and paired spins), and the
# eigensolver gives itself some breathing room by computing some extra
# states (see `n_ep_extra` argument to `self_consistent_field`).
#
# We can check the occupations ...
hcat(scfres.occupation...)
# ... and density, where we use that the density objects in DFTK are
# indexed as ρ[iσ, ix, iy, iz], i.e. first in the spin component and then
# in the 3-dimensional real-space grid.
rvecs = collect(r_vectors(basis))[:, 1, 1]  # slice along the x axis
x = [r[1] for r in rvecs]                   # only keep the x coordinate
plot(x, scfres.ρ[1, :, 1, 1], label="", xlabel="x", ylabel="ρ", marker=2)

# We can also perform various postprocessing steps:
# for instance compute a band structure
plot_bandstructure(scfres, kline_density=5)
# or get the cartesian forces (in Hartree / Bohr)
compute_forces_cart(scfres)[1]  # Select silicon forces
# The `[1]` extracts the forces for the first kind of atoms,
# i.e. `Si` (silicon) in the setup of the `atoms` list of step 1 above.
# As expected, they are almost zero in this highly symmetric configuration.

# ## Where to go from here
# Take a look at the
#md # [example index](@ref example-index)
#nb # [example index](https://juliamolsim.github.io/DFTK.jl/dev/#example-index-1)
# to continue exploring DFTK.
