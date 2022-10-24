# # Comparison of DFT solvers

# We compare four different approaches for solving the DFT minimisation problem,
# namely a density-based SCF, a potential-based SCF, direct minimisation and Newton.

# First we setup our problem
using DFTK
using LinearAlgebra

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms     = [Si, Si]
positions = [ones(3)/8, -ones(3)/8]

model = model_LDA(lattice, atoms, positions)
basis = PlaneWaveBasis(model; Ecut=5, kgrid=[3, 3, 3])

## Convergence we desire
tol = 1e-12
is_converged = DFTK.ScfConvergenceDensity(tol);

# ## Density-based self-consistent field
scfres_scf = self_consistent_field(basis; is_converged);

# ## Potential-based SCF
scfres_scfv = DFTK.scf_potential_mixing(basis; is_converged);

# ## Direct minimization
scfres_dm = direct_minimization(basis; tol);

# ## Newton algorithm

# Start not too far from the solution to ensure convergence:
# We run first a very crude SCF to get close and then switch to Newton.
scfres_start = self_consistent_field(basis; tol=1e-1);

# Remove the virtual orbitals (which Newton cannot treat yet)
ψ = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ
scfres_newton = newton(basis, ψ; tol);

# ## Comparison of results

println("|ρ_newton - ρ_scf|  = ", norm(scfres_newton.ρ - scfres_scf.ρ))
println("|ρ_newton - ρ_scfv| = ", norm(scfres_newton.ρ - scfres_scfv.ρ))
println("|ρ_newton - ρ_dm|   = ", norm(scfres_newton.ρ - scfres_dm.ρ))
