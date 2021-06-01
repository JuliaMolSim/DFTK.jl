# Very basic setup, useful for testing
using DFTK, PyPlot

a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

model = model_LDA(lattice, atoms)
kgrid = [2, 2, 2]  # k-point grid (Regular Monkhorst-Pack grid)
Ecut = 30           # kinetic energy cutoff in Hartree
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
tol = 1e-12

scfres_scf = self_consistent_field(basis, tol=tol,
                                   is_converged=DFTK.ScfConvergenceDensity(tol))
scfres_test = self_consistent_field(basis, tol=tol, maxiter=5,
                                   is_converged=DFTK.ScfConvergenceDensity(tol))

## start no too far from the solution to ensure convergence of newton algorithm
scfres = newton(basis, ψ0=scfres_test.ψ, tol=tol, max_iter=20)

## plot densities from both algorithms
rvecs = collect(r_vectors(basis))[:, 1, 1]  # slice along the x axis
x = [r[1] for r in rvecs]                   # only keep the x coordinate
figure()
plot(x, scfres.ρ[:, 1, 1], label="newton")
plot(x, scfres_scf.ρ[:, 1, 1], label="scf")
println("|ρ_newton - ρ_scf| = ", norm(scfres.ρ - scfres_scf.ρ))
