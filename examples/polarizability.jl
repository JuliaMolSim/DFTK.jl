# # Polarizability by linear response
#
# We compute the polarizability of a Helium atom. The polarizability
# is defined as the change in dipole moment
# ```math
# d = \int r ρ(r) dr
# ```
# with respect to a small uniform electric field ``E = -x``.
#
# We compute this in two ways: first by finite differences (applying a
# finite electric field), then by linear response. Note that DFTK is
# not really adapted to isolated atoms because it uses periodic
# boundary conditions. Nevertheless we can simply embed the Helium
# atom in a large enough box (although this is computationally wasteful).
#
# As in other tests, this is not fully converged, convergence
# parameters were simply selected for fast execution on CI,

using DFTK
using PyPlot
using KrylovKit

a = 10.
lattice = a * I(3)  # cube of ``a`` bohrs
He = ElementPsp(:He, psp=load_psp("hgh/lda/He-q2"))
atoms = [He => [[1/2; 1/2; 1/2]]]  # Helium at the center of the box

kgrid = [1, 1, 1]  # no kpoint sampling for an isolated system
Ecut = 30
tol = 1e-8

## dipole moment of a given density
function dipole(ρ)
    basis = ρ.basis
    rr = [a*(r[1]-1/2) for r in r_vectors(basis)]
    d = sum(rr .* ρ.real) * basis.model.unit_cell_volume / prod(basis.fft_size)
end

# We first compute the polarizability by finite differences: compute
# the density at rest, and then in a small uniform electric field.
model = model_LDA(lattice, atoms; symmetry=:off)
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
res = self_consistent_field(basis, tol=tol)
dref = dipole(res.ρ)

ε = .01
model_ε = model_LDA(lattice, atoms; extra_terms=[ExternalFromReal(r -> -ε*(r[1]-a/2))], symmetry=:off)
basis_ε = PlaneWaveBasis(model_ε, Ecut; kgrid=kgrid)
res_ε = self_consistent_field(basis_ε, tol=tol)
dε = dipole(res_ε.ρ)

println("Reference dipole: $dref")
println("Displaced dipole : $dε")
println("Polarizability : $((dε-dref)/ε)")

# The result on more converged grids is very close to published results (https://pubs.rsc.org/en/content/articlelanding/2018/cp/c8cp03569e): 1.65 with LSDA (1.38 with CCSD(T)).
#
# Now we use linear response to compute this analytically; we refer to standard textbooks for the formalism. In the following, ``\chi_0`` is the independent-particle polarizability, and ``K`` the Hartree-exchange-correlation kernel
# ```math
# \begin{aligned}
# d\rho &= \chi_0 dV\\
#       &= \chi_0 (dV_{\rm ext} + K d\rho)\\
#       &= (1-\chi_0 K)^-1 \chi_0 dV_{\rm ext}
# \end{aligned}
# ```
# so the polarizability operator is ``\chi = (1-\chi_0 K)^-1 \chi_0``. Numerically, we apply ``\chi`` to ``dV = -x`` by solving a linear equation (the Dyson equation) iteratively.

devec(arr) = from_real(basis, reshape(arr, size(res.ρ.real)))

## apply χ0 K to a vectorized dρ
function eps_fun(dρ)
    dρ = devec(dρ)
    dv = apply_kernel(basis, dρ; ρ=res.ρ)
    χ0dv = apply_χ0(res.ham, res.ψ, res.εF, res.eigenvalues, dv)
    vec((dρ - χ0dv).real)
end

dVext = from_real(basis, [-a*(r[1]-1/2) for r in r_vectors(basis)])
dρ_nointeract = apply_χ0(res.ham, res.ψ, res.εF, res.eigenvalues, dVext)
dρ = devec(linsolve(eps_fun, vec(dρ_nointeract.real), verbosity=3)[1])

println("Non-interacting polarizability: $(dipole(dρ_nointeract))")
println("Interacting polarizability: $(dipole(dρ))")

# As expected, the interacting polarizability matches the finite difference result. The non-interacting polarizability is higher.
