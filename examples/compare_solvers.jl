## We compare here three different solvers : SCF, direct minimization and newton
## algorithm.

using DFTK, LinearAlgebra

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

model = model_LDA(lattice, atoms)
basis = PlaneWaveBasis(model; Ecut=5, kgrid=[3, 3, 3])
tol = 1e-12

## SCF
scfres_scf = self_consistent_field(basis, tol=tol,
                                   is_converged=DFTK.ScfConvergenceDensity(tol))

## Direct minimization
scfres_dm = direct_minimization(basis, tol=tol)

## Newton algorithm
# start not too far from the solution to ensure convergence : we use here the
# solution of a single iteration SCF
scfres_start = self_consistent_field(basis, maxiter=1)
# remove virtual orbitals
ψ = DFTK.select_occupied_orbitals(basis, scfres_start.ψ)
scfres_newton = newton(basis, ψ, tol=tol)

println("|ρ_newton - ρ_scf| = ", norm(scfres_newton.ρ - scfres_scf.ρ))
println("|ρ_newton - ρ_dm| = ", norm(scfres_newton.ρ - scfres_dm.ρ))
println("|ρ_scf - ρ_dm| = ", norm(scfres_scf.ρ - scfres_dm.ρ))
