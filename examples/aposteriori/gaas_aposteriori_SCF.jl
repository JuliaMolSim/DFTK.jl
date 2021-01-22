# This file contains test of the estimator
#
# P-P* = (Ω+K)^{-1}[P,[P,H(P)]]
#
# and
#
# M^1/2(P-P*) = M^1/2(Ω+K)^{-1}M^1/2 * M^-1/2[P,[P,H(P)]]
#
# translated to orbitals, in the linear case.
# We look at is the basis error : φ* is computed for a reference
# Ecut_ref and then we measure the error φ-φ* and the residual obtained for
# smaller Ecut
#

using DFTK
using LinearAlgebra
using PyPlot

# import aux files
include("aposteriori_tools.jl")
include("aposteriori_callback.jl")

# Very basic setup, useful for testing
# model parameters
a = 10.68290949909  # GaAs lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Ga = ElementPsp(:Ga, psp=load_psp("hgh/lda/ga-q3"))
As = ElementPsp(:As, psp=load_psp("hgh/lda/as-q5"))
atoms = [Ga => [ones(3)/8], As => [-ones(3)/8]]

## local potential only
model = model_LDA(lattice, atoms)

kgrid = [1, 1, 1]   # k-point grid (Regular Monkhorst-Pack grid)
tol = 1e-10
tol_krylov = 1e-12
Ecut_ref = 15           # kinetic energy cutoff in Hartree

## changing norm for error estimation
change_norm = true

println("--------------------------------")
println("reference computation")
basis_ref = PlaneWaveBasis(model, Ecut_ref; kgrid=kgrid)
scfres_ref = self_consistent_field(basis_ref, tol=tol,
                                   determine_diagtol=DFTK.ScfDiagtol(diagtol_max=1e-10),
                                   is_converged=DFTK.ScfConvergenceDensity(tol),
                                   callback=callback_estimators())
