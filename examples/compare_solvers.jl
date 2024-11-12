# # Comparison of DFT solvers

# We compare four different approaches for solving the DFT minimisation problem,
# namely a density-based SCF, a potential-based SCF, direct minimisation and Newton.

# First we setup our problem
using AtomsBuilder
using DFTK
using LinearAlgebra

system = attach_psp(bulk(:Si); Si="hgh/lda/Si-q4")
model  = model_DFT(system; functionals=PBEsol())
basis  = PlaneWaveBasis(model; Ecut=5, kgrid=[3, 3, 3])

## Convergence we desire in the density
tol = 1e-6

# ## Density-based self-consistent field
scfres_scf = self_consistent_field(basis; tol);

# ## Potential-based SCF
scfres_scfv = DFTK.scf_potential_mixing(basis; tol);

# ## Direct minimization
scfres_dm = direct_minimization(basis; tol);

# ## Newton algorithm

# Start not too far from the solution to ensure convergence:
# We run first a very crude SCF to get close and then switch to Newton.
scfres_start = self_consistent_field(basis; tol=0.5);

# Remove the virtual orbitals (which Newton cannot treat yet)
ψ = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ
scfres_newton = newton(basis, ψ; tol);

# ## Comparison of results

println("|ρ_newton - ρ_scf|  = ", norm(scfres_newton.ρ - scfres_scf.ρ))
println("|ρ_newton - ρ_scfv| = ", norm(scfres_newton.ρ - scfres_scfv.ρ))
println("|ρ_newton - ρ_dm|   = ", norm(scfres_newton.ρ - scfres_dm.ρ))
