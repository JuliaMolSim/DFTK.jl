# This file contains test of the estimator
#
# P-P* = (Ω+K)^{-1}[P,[P,H(P)]]
#
# when the error we look at is the SCF error : P* is computed as the converged
# density matrix and then we measure the error P-P* and the residual along the
# iterations. To this end, we defined a custom callback function
# (currently, only Nk = 1 kpt only is supported)
#
#            !!! NOT OPTIMIZED YET !!!
#

# Very basic setup, useful for testing
using DFTK
using LinearAlgebra
using PyPlot

# model parameters
a = 10.68290949909  # GaAs lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Ga = ElementPsp(:Ga, psp=load_psp("hgh/lda/ga-q3"))
As = ElementPsp(:As, psp=load_psp("hgh/lda/as-q5"))
atoms = [Ga => [ones(3)/8], As => [-ones(3)/8]]

# define different models
modelLDA = model_LDA(lattice, atoms)
kgrid = [1, 1, 1]   # k-point grid (Regular Monkhorst-Pack grid)
tol = 1e-12
tol_krylov = 1e-15
Ecut = 10           # kinetic energy cutoff in Hartree

basis_scf = PlaneWaveBasis(modelLDA, Ecut; kgrid=kgrid)

# import aux file
include("aposteriori_operators.jl")
include("aposteriori_callback.jl")
include("newton.jl")

ite = nothing
φ_list = nothing


scfres = self_consistent_field(basis_scf, tol=tol,
                               is_converged=DFTK.ScfConvergenceDensity(tol),
                               determine_diagtol=DFTK.ScfDiagtol(diagtol_max=1e-14),
                               callback=callback_estimators(test_newton=false, change_norm=true))
