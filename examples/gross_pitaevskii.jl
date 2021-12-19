# # Gross-Pitaevskii equation in one dimension
# In this example we will use DFTK to solve
# the Gross-Pitaevskii equation, and use this opportunity to explore a few internals.

# ## The model
# The [Gross-Pitaevskii equation](https://en.wikipedia.org/wiki/Gross%E2%80%93Pitaevskii_equation) (GPE)
# is a simple non-linear equation used to model bosonic systems
# in a mean-field approach. Denoting by ``ψ`` the effective one-particle bosonic
# wave function, the time-independent GPE reads in atomic units:
# ```math
#     H ψ = \left(-\frac12 Δ + V + 2 C |ψ|^2\right) ψ = μ ψ \qquad \|ψ\|_{L^2} = 1
# ```
# where ``C`` provides the strength of the boson-boson coupling.
# It's in particular a favorite model of applied mathematicians because it
# has a structure simpler than but similar to that of DFT, and displays
# interesting behavior (especially in higher dimensions with magnetic fields, see
# [Gross-Pitaevskii equation with external magnetic field](@ref)).

# We wish to model this equation in 1D using DFTK.
# First we set up the lattice. For a 1D case we supply two zero lattice vectors,
a = 10
lattice = a .* [[1 0 0.]; [0 0 0]; [0 0 0]];

# which is special cased in DFTK to support 1D models.
#
# For the potential term `V` we just pick a harmonic
# potential. The real-space grid is in ``[0,1)``
# in fractional coordinates( see
# [Lattices and lattice vectors](@ref conventions-lattice)),
# therefore:
pot(x) = (x - a/2)^2;

# We setup each energy term in sequence: kinetic, potential and nonlinear term.
# For the non-linearity we use the `PowerNonlinearity(C, α)` term of DFTK.
# This object introduces an energy term ``C ∫ ρ(r)^α dr``
# to the total energy functional, thus a potential term ``α C ρ^{α-1}``.
# In our case we thus need the parameters
C = 1.0
α = 2;

# ... and with this build the model
using DFTK
using LinearAlgebra

n_electrons = 1  # Increase this for fun
terms = [Kinetic(),
         ExternalFromReal(r -> pot(r[1])),
         PowerNonlinearity(C, α),
]
model = Model(lattice; n_electrons=n_electrons, terms=terms,
              spin_polarization=:spinless);  # use "spinless electrons"

# We discretize using a moderate Ecut (For 1D values up to `5000` are completely fine)
# and run a direct minimization algorithm:
basis = PlaneWaveBasis(model, Ecut=500, kgrid=(1, 1, 1))
scfres = direct_minimization(basis, tol=1e-8) # This is a constrained preconditioned LBFGS
scfres.energies

# ## Internals
# We use the opportunity to explore some of DFTK internals.
#
# Extract the converged density and the obtained wave function:
ρ = real(scfres.ρ)[:, 1, 1, 1]  # converged density, first spin component
ψ_fourier = scfres.ψ[1][:, 1];    # first k-point, all G components, first eigenvector

# Transform the wave function to real space and fix the phase:
ψ = G_to_r(basis, basis.kpoints[1], ψ_fourier)[:, 1, 1]
ψ /= (ψ[div(end, 2)] / abs(ψ[div(end, 2)]));

# Check whether ``ψ`` is normalised:
x = a * vec(first.(DFTK.r_vectors(basis)))
N = length(x)
dx = a / N  # real-space grid spacing
@assert sum(abs2.(ψ)) * dx ≈ 1.0

# The density is simply built from ψ:
norm(scfres.ρ - abs2.(ψ))

# We summarize the ground state in a nice plot:
using Plots

p = plot(x, real.(ψ), label="real(ψ)")
plot!(p, x, imag.(ψ), label="imag(ψ)")
plot!(p, x, ρ, label="ρ")

# The `energy_hamiltonian` function can be used to get the energy and
# effective Hamiltonian (derivative of the energy with respect to the density matrix)
# of a particular state (ψ, occupation).
# The density ρ associated to this state is precomputed
# and passed to the routine as an optimization.
E, ham = energy_hamiltonian(basis, scfres.ψ, scfres.occupation; ρ=scfres.ρ)
@assert E.total == scfres.energies.total

# Now the Hamiltonian contains all the blocks corresponding to k-points. Here, we just have one k-point:
H = ham.blocks[1];

# `H` can be used as a linear operator (efficiently using FFTs), or converted to a dense matrix:
ψ11 = scfres.ψ[1][:, 1] # first k-point, first eigenvector
Hmat = Array(H) # This is now just a plain Julia matrix, 
##                which we can compute and store in this simple 1D example
@assert norm(Hmat * ψ11 - H * ψ11) < 1e-10

# Let's check that ψ11 is indeed an eigenstate:
norm(H * ψ11 - dot(ψ11, H * ψ11) * ψ11)

# Build a finite-differences version of the GPE operator ``H``, as a sanity check:
A = Array(Tridiagonal(-ones(N - 1), 2ones(N), -ones(N - 1)))
A[1, end] = A[end, 1] = -1
K = A / dx^2 / 2
V = Diagonal(pot.(x) + C .* α .* (ρ.^(α-1)))
H_findiff = K + V;
maximum(abs.(H_findiff*ψ - (dot(ψ, H_findiff*ψ) / dot(ψ, ψ)) * ψ))
