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
# is assumed throughout.
#
# !!! note "Convergence parameters in the documentation"
#     We use rough parameters in order to be able
#     to automatically generate this documentation very quickly.
#     Therefore results are far from converged.
#     Tighter thresholds and larger grids should be used for more realistic results.
#
# For our discussion we will use the classic example of
# computing the LDA ground state of the silicon crystal.
# Performing such a calculation roughly proceeds in three steps.

using DFTK
using Plots

## 1. Define lattice and atomic positions
a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]

## Load HGH pseudopotential for Silicon
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))

## Specify type and positions of atoms
atoms = [Si => [ones(3)/8, -ones(3)/8]]

## 2. Select model and basis
model = model_LDA(lattice, atoms)
kgrid = [4, 4, 4]  # k-point grid (Regular Monkhorst-Pack grid)
Ecut = 7           # kinetic energy cutoff in Hartree
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)

## 3. Run the SCF procedure to obtain the ground state
scfres = self_consistent_field(basis, tol=1e-8);

# Note that DFTK by default applies the convergence tolerance `tol`
# to the energy difference, so that the norm in the density difference
# is not yet converged to 8 digits.
#
# That's it! Now you can get various quantities from the result of the SCF.
# For instance, the energies:
scfres.energies
# Eigenvalues: here `scfres.eigenvalues` returns a 7x8 Array where 7 is the
# number of eigenvalues that are computed and 8 the number of kpoints.
hcat(scfres.eigenvalues...)
# Use `scfres.occupation` to see the number of electrons on each energy level.
hcat(scfres.occupation...)
# And density:
rvecs = collect(r_vectors(basis))[:, 1, 1]  # slice along the x axis
x = [r[1] for r in rvecs]                   # only keep the x coordinate
plot(x, scfres.ρ.real[:, 1, 1], label="", xlabel="x", ylabel="ρ", marker=2)

# We can also perform various postprocessing steps:
# for instance compute a band structure
n_bands = 8
plot_bandstructure(scfres, n_bands, kline_density=5, unit=:eV)
# or forces
forces(scfres)[1]  # Select silicon forces
# The `[1]` extracts the forces for the first kind of atoms,
# i.e. `Si` (silicon) in the setup of the `atoms` list of step 1 above.
# As expected, they are almost zero in this highly symmetric configuration.

# ## Where to go from here
# Take a look at the
#md # [example index](@ref example-index)
#nb # [example index](https://juliamolsim.github.io/DFTK.jl/dev/#example-index-1)
# to continue exploring DFTK.
