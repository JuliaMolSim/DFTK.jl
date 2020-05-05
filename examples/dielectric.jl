# Compute the full dielectric matrix (q=0,ω=0)

using DFTK
using Plots
using KrylovKit

# Calculation parameters
kgrid = [1, 1, 1]       # k-Point grid
Ecut = 1               # kinetic energy cutoff in Hartree

# Setup silicon lattice
a = 10.263141334305942  # Silicon lattice constant in Bohr
lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

model = model_LDA(lattice, atoms)
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
scfres = self_consistent_field(basis, tol=1e-14)

chi0 = compute_χ0(scfres.ham)
Kh = DFTK.compute_hartree_kernel(basis)
Kxc = DFTK.compute_xc_kernel(basis, scfres.ρ.real)

eps = I - (Kh+Kxc)*chi0
res = eigsolve(eps, length(scfres.ρ.real), 3, :SR)

function epsfun(dv)
    dρ = apply_χ0(scfres.ham, reshape(dv, size(scfres.ρ.real)), scfres.ψ, scfres.εF, scfres.eigenvalues)
    Kdρ = DFTK.apply_hartree_kernel(basis, dρ) + DFTK.apply_xc_kernel(basis, scfres.ρ.real, dρ)
    dv - vec(Kdρ)
end

e1, v1 = eigsolve(epsfun, length(scfres.ρ.real), 3, :SR)
e2, v2 = eigsolve(eps, length(scfres.ρ.real), 3, :SR)
