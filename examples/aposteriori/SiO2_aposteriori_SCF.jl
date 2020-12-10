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
using PyCall
using DFTK
using LinearAlgebra
using PyPlot

# import aux file
include("aposteriori_operators.jl")
include("newton.jl")

# Lattice parameters for α-quartz from
# https://www.materialsproject.org/materials/mp-549166/
# All lengths in Ångström (used by ASE)
a = b = 4.98530291
c = 5.47852971
α = β = 90
γ = 120

silicon = [  # Fractional coordinates
           [0.000000  0.471138  0.833333],
           [0.471138  0.000000  0.166666],
           [0.528862  0.528862  0.500000],
          ]
oxygen = [  # Fractional coordinates
          [0.148541  0.734505  0.046133],
          [0.265495  0.414036  0.712800],
          [0.585964  0.851459  0.379466],
          [0.734505  0.148541  0.953867],
          [0.414036  0.265495  0.287200],
          [0.851459  0.585964  0.620534],
         ]
symbols = append!(fill("Si", 3), fill("O", 6))
atoms_ase = pyimport("ase").Atoms(symbols=symbols, cell=[a, b, c, α, β, γ], pbc=true,
                                  scaled_positions=vcat(vcat(silicon, oxygen)...))

atoms = load_atoms(atoms_ase)
atoms = [ElementPsp(el.symbol, psp=load_psp(el.symbol, functional="pbe")) => position
         for (el, position) in atoms]

lattice = load_lattice(atoms_ase)

# define different models
modelLDA = model_LDA(lattice, atoms)
kgrid = [1, 1, 1]   # k-point grid (Regular Monkhorst-Pack grid)
tol = 1e-12
tol_krylov = 1e-15
Ecut = 10           # kinetic energy cutoff in Hartree

ite = nothing
φ_list = nothing
basis_list = nothing

basis_scf = PlaneWaveBasis(modelLDA, Ecut; kgrid=kgrid)

scfres = self_consistent_field(basis_scf, tol=tol,
                               is_converged=DFTK.ScfConvergenceDensity(tol),
                               #  determine_diagtol=DFTK.ScfDiagtol(diagtol_max=1e-12),
                               callback=callback_estimators())
