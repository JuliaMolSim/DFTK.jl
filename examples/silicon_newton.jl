# Very basic setup, useful for testing
using DFTK
using LinearAlgebra
using PyPlot

# import aux file
include("aposteriori_operators.jl")
include("newton.jl")

# model parameters
a = 10.26  # Silicon lattice constant in Bohr
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

mod = model_LDA(lattice, atoms, n_electrons=2)
kgrid = [1, 1, 1]   # k-point grid (Regular Monkhorst-Pack grid)
Ecut = 15           # kinetic energy cutoff in Hartree
basis_scf = PlaneWaveBasis(mod, Ecut; kgrid=kgrid)

tol = 1e-12
scfres = self_consistent_field(basis_scf, tol=tol,
                               is_converged=DFTK.ScfConvergenceDensity(tol))

## testing newton starting not far away from the SCF solution
ψ0 = deepcopy(scfres.ψ)
for ik = 1:length(ψ0)
    ψ0k = ψ0[ik]
    for i in 1:4
        ψ0k[:,i] += randn(size(ψ0k[:,i]))*1e-2
    end
end
φ0 = similar(ψ0)
for ik = 1:length(φ0)
    φ0[ik] = ψ0[ik][:,1:1]
    φ0[ik] = Matrix(qr(φ0[ik]).Q)
end
newton(basis_scf; ψ0=φ0, tol=1e-12)
