## Gross-Pitaevskii equation: -1/2 Δ ψ + V ψ + 2C |ψ|^2 ψ = λ ψ, ||ψ||_L^2 = 1

using DFTK
using Plots
using LinearAlgebra

Ecut = 4000
# Nonlinearity : energy C ∫ρ^α
C = 1.0
α = 2

# Unit cell. Having two lattice vectors as zero means a 1D system
a = 10
lattice = a .* [[1 0 0.]; [0 0 0]; [0 0 0]]

f(x) = (x-a/2)^2 # potential

n_electrons = 1  # increase this for fun
# We add the needed terms
terms = [Kinetic(),
         ExternalFromReal(X -> f(X[1])),
         PowerNonlinearity(C, α),
]
model = Model(lattice; n_electrons=n_electrons, terms=terms,
              spin_polarization=:spinless)  # "spinless fermions"
basis = PlaneWaveBasis(model, Ecut)

scfres = direct_minimization(basis, x_tol=1e-8, f_tol=-1, g_tol=-1)
println()
display(scfres.energies)

x = a * range(0, 1, length=basis.fft_size[1]+1)[1:end-1]
ρ = real(scfres.ρ.real)[:, 1, 1] # converged density
ψ_fourier = scfres.ψ[1][:, 1] # first kpoint, all G components, first eigenvector
ψ = G_to_r(basis, basis.kpoints[1], ψ_fourier)[:, 1, 1] # IFFT back to real space
@assert sum(abs2.(ψ)) * (x[2]-x[1]) ≈ 1.0

# phase fix
ψ /= (ψ[div(end, 2)] / abs(ψ[div(end, 2)]))

# ψ solves -Δ/2 ψ + Vext ψ + C α ρ^(α-1) ψ = λ ψ. Check that with finite differences
N = length(x)
A = Array(Tridiagonal(-ones(N-1), 2ones(N), -ones(N-1)))
A[1, end] = A[end, 1] = -1
K = A / ((x[2]-x[1])^2) / 2
V = Diagonal(f.(x) + C .* α .* (ρ.^(α-1)))
H = K+V

p = plot(x, real.(ψ), label="ψreal")
plot!(p, x, imag.(ψ), label="ψimag")
plot!(p, x, ρ, label="ρ")
plot!(p, x, abs.(H*ψ - dot(ψ, H*ψ)/dot(ψ, ψ)*ψ), label="residual")
gui(p)
