# # Gross-Pitaevskii equation
#
# In this example we will use DFTK to solve
# the [Gross-Pitaevskii equation](https://en.wikipedia.org/wiki/Gross%E2%80%93Pitaevskii_equation) (GPE),
# which is a simple non-linear equation to model bosonic systems
# in a mean-field approach. Denoting by ``ψ`` the effective one-particle bosonic
# wave function, the time-independent GPE reads in atomic units:
# ```math
#     H ψ = (-\frac12 Δ + V + 2 C |ψ|^2) ψ = μ ψ \qquad \|ψ\|_{L^2} = 1
# ```
# where ``C`` is related to the boson-boson `s`-wave scattering length and provides the
# strength of the boson-boson coupling and `μ` (chemical potential) is the lowest eigenvalue
# of the associated non-linear Hermitian operator.

# We wish to model this equation in 1D using DFTK.
# First we set up the lattice. For a 1D case we supply two zero lattice vectors:
a = 10
lattice = a .* [[1 0 0.]; [0 0 0]; [0 0 0]];

# Next we setup the GPE model. For the potential term V`
# we just pick a Harmonic potential:
f(x) = (x - a/2)^2;

# For the non-linearity we use the `PowerNonlinearity(C, α)` term of DFTK.
# This object introduces an energy term ``C ∫ (ρ(r))^α dr``
# to the total energy functional, thus a potential term ``α C ρ^(α-1)``
# (just the derivative wrt. ``ρ``). We therefore need the parameters
C = 1.0
α = 2;

# ... and finally build the model
using DFTK
using LinearAlgebra

n_electrons = 1  # Increase this for fun
terms = [Kinetic(),
         ExternalFromReal(r -> f(norm(r))),
         PowerNonlinearity(C, α),
]
model = Model(lattice; n_electrons=n_electrons, terms=terms,
              spin_polarization=:spinless);  # use "spinless fermions"

# We discretize using a moderate Ecut (For 1D values up to `5000` are completely fine)
# and run direct minimization:
Ecut = 500
basis = PlaneWaveBasis(model, Ecut)
scfres = direct_minimization(basis, tol=1e-8);
#-
scfres.energies

# Extract the converged density and the obtained one-particle wave function
ρ = real(scfres.ρ.real)[:, 1, 1]  # converged density
ψ_fourier = scfres.ψ[1][:, 1];    # first kpoint, all G components, first eigenvector

# Transform the wave function to real space and fix the phase:
ψ = G_to_r(basis, basis.kpoints[1], ψ_fourier)[:, 1, 1]
ψ /= (ψ[div(end, 2)] / abs(ψ[div(end, 2)]));

# Check whether ``ψ`` is normalised:
x = a * vec(first.(DFTK.r_vectors(basis)))
N = length(x)
dx = a / N  # real-space grid spacing
@assert sum(abs2.(ψ)) * dx ≈ 1.0

# Build a finite-differences version of the GPE operator ``H``:
A = Array(Tridiagonal(-ones(N - 1), 2ones(N), -ones(N - 1)))
A[1, end] = A[end, 1] = -1
K = A / dx^2 / 2
V = Diagonal(f.(x) + C .* α .* (ρ.^(α-1)))
H = K+V;

# Summarising the ground state in a nice plot:
using Plots

p = plot(x, real.(ψ), label="real(ψ)")
plot!(p, x, imag.(ψ), label="imag(ψ)")
plot!(p, x, ρ, label="ρ")
plot!(p, x, abs.(H*ψ - (dot(ψ, H*ψ) / dot(ψ, ψ)) * ψ), label="residual")
