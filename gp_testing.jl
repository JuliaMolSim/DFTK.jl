a = 10
lattice = a .* [[1 0 0.]; [0 0 0]; [0 0 0]];
pot(x) = (x - a/2)^2;
C = 1.0
α = 2;

using DFTK
using LinearAlgebra

n_electrons = 1  # Increase this for fun
terms = [Kinetic(),
         ExternalFromReal(r -> pot(r[1])),
         LocalNonlinearity(ρ -> C * ρ^α),
]
model = Model(lattice; n_electrons, terms, spin_polarization=:spinless);  # spinless electrons

# We discretize using a moderate Ecut (For 1D values up to `5000` are completely fine)
# and run a direct minimization algorithm:
basis = PlaneWaveBasis(model, Ecut=500, kgrid=(1, 1, 1))
scfres = self_consistent_field(basis; tol=1e-8, mixing=SimpleMixing()) # This is a constrained preconditioned LBFGS
scfres.energies

# ## Internals
# We use the opportunity to explore some of DFTK internals.
#
# Extract the converged density and the obtained wave function:
ρ = real(scfres.ρ)[:, 1, 1, 1]  # converged density, first spin component
ψ_fourier = scfres.ψ[1][:, 1];    # first k-point, all G components, first eigenvector

# Transform the wave function to real space and fix the phase:
ψ = ifft(basis, basis.kpoints[1], ψ_fourier)[:, 1, 1]
ψ /= (ψ[div(end, 2)] / abs(ψ[div(end, 2)]));

# Check whether ``ψ`` is normalised:
x = a * vec(first.(DFTK.r_vectors(basis)))
N = length(x)
dx = a / N  # real-space grid spacing
@assert sum(abs2.(ψ)) * dx ≈ 1.0

# The density is simply built from ψ:
norm(scfres.ρ - abs2.(ψ))

nothing