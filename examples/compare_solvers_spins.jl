## We compare here three different solvers : SCF, direct minimization and newton
## algorithm.

using DFTK

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]
m = 1
magnetic_moments = [Si => [m, m]]

model = model_LDA(lattice, atoms, magnetic_moments=magnetic_moments)
kgrid = [1, 1, 1]  # k-point grid (Regular Monkhorst-Pack grid)
Ecut = 5           # kinetic energy cutoff in Hartree
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
tol = 1e-12

## SCF
println("---------------------------------------------")
println("SCF")
ρ0 = guess_density(basis, magnetic_moments)
scfres_scf = self_consistent_field(basis, tol=tol, ρ=ρ0,
                                   is_converged=DFTK.ScfConvergenceDensity(tol))

## Direct minimization
println("---------------------------------------------")
println("Direct minimization")
scfres_dm = direct_minimization(basis, tol=tol)

## Newton algorithm
# start not too far from the solution to ensure convergence : we use here the
# solution of a 1 iteration SCF cycle
println("---------------------------------------------")
println("Newton algorithm")
scfres_start = self_consistent_field(basis, tol=tol, ρ=ρ0, maxiter=1,
                                   is_converged=DFTK.ScfConvergenceDensity(tol))
scfres_newton = newton(basis, ψ0=scfres_start.ψ, tol=tol)

println("---------------------------------------------")
println("|ρ_newton - ρ_scf| = ", norm(scfres_newton.ρ - scfres_scf.ρ))
println("|ρ_newton - ρ_dm| = ", norm(scfres_newton.ρ - scfres_dm.ρ))
println("|ρ_scf - ρ_dm| = ", norm(scfres_scf.ρ - scfres_dm.ρ))
