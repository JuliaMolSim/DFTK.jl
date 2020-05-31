# # 1D Gross-Pitaevskii equation
#
# In this example we will use DFTK to solve
# the [Gross-Pitaevskii equation](https://en.wikipedia.org/wiki/Gross%E2%80%93Pitaevskii_equation) (GPE),
# which is a simple non-linear equation to model bosonic systems
# in a mean-field approach. Denoting by ``ψ`` the effective one-particle bosonic
# wave function, the time-independent GPE reads in atomic units:
# ```math
#     H ψ = \left(-\frac12 Δ + V + 2 C |ψ|^2\right) ψ = μ ψ \qquad \|ψ\|_{L^2} = 1
# ```
# where ``C``  provides the
# strength of the boson-boson coupling.

# We wish to model this equation in 1D using DFTK.
# First we set up the lattice. For a 1D case we supply two zero lattice vectors:
a = 10
lattice = a .* [[1 0 0.]; [0 0 0]; [0 0 0]];

# which is special cased in DFTK to support 1D models.
#
# For the potential term V` we just pick a Harmonic
# potential (the grid is ``[0,1)`` in fractional coordinates, see
#md # [Lattices and lattice vectors](@ref)
#nb # [Lattices and lattice vectors](https://juliamolsim.github.io/DFTK.jl/dev/advanced/conventions.html#conventions-lattice-1)
# )
pot(x) = (x - a/2)^2;

# Set the parameters for the nonlinear term:
C = 1.0
α = 2;

# ... and finally build the model
using DFTK
using LinearAlgebra
n_electrons = 1  # Increase this for fun

# We setup each energy term in sequence: kinetic, potential and nonlinear term.
# For the non-linearity we use the `PowerNonlinearity(C, α)` term of DFTK.
# This object introduces an energy term ``C ∫ ρ(r)^α dr``
# to the total energy functional, thus a potential term ``α C ρ^{α-1}``.
terms = [Kinetic(),
         ExternalFromReal(r -> pot(norm(r))),
         PowerNonlinearity(C, α),
]
model = Model(lattice; n_electrons=n_electrons, terms=terms,
              spin_polarization=:spinless);  # use "spinless electrons"

# We discretize using a moderate Ecut (For 1D values up to `5000` are completely fine)
# and run a direct minimization algorithm:
Ecut = 500
basis = PlaneWaveBasis(model, Ecut)
scfres = direct_minimization(basis, tol=1e-8); # This is a constrained preconditioned LBFGS
#-
scfres.energies

# Extract the converged density and the obtained wave function
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

# Build a finite-differences version of the GPE operator ``H``, as a sanity check:
A = Array(Tridiagonal(-ones(N - 1), 2ones(N), -ones(N - 1)))
A[1, end] = A[end, 1] = -1
K = A / dx^2 / 2
V = Diagonal(pot.(x) + C .* α .* (ρ.^(α-1)))
H = K+V;

# We summarize the ground state in a nice plot, and check that our
# spectral method agrees with the finite difference discretization:
using Plots

p = plot(x, real.(ψ), label="real(ψ)")
plot!(p, x, imag.(ψ), label="imag(ψ)")
plot!(p, x, ρ, label="ρ")
plot!(p, x, abs.(H*ψ - (dot(ψ, H*ψ) / dot(ψ, ψ)) * ψ), label="residual")
