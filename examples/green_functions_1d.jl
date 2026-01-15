# Example: Computing Periodic Green's Functions in 1D
# 
# This example demonstrates the periodic Green's function computation 
# framework for a 1D harmonic oscillator system.

using DFTK
using LinearAlgebra

# Setup 1D system 
a = 10.0  # Box size in atomic units
lattice = a .* [[1 0 0.]; [0 0 0]; [0 0 0]]  # 1D lattice

# Periodic cosine potential V(x) = cos(2π x / a)
potential(x) = cos(2π * x / a)

# Build model (simple kinetic + external potential, spinless)
# Disable symmetry at model level for Green's function computation
n_electrons = 1
terms = [
    Kinetic(),
    ExternalFromReal(r -> potential(r[1])),
]
model = Model(lattice; n_electrons, terms, spin_polarization=:spinless, symmetries=false)

# Create plane-wave basis
Ecut = 50
kgrid = MonkhorstPack([4, 1, 1])  # 4 k-points in 1D
basis = PlaneWaveBasis(model; Ecut, kgrid)

println("Basis setup:")
println("  Grid size: ", basis.fft_size)
println("  Number of k-points: ", length(basis.kpoints))
println("  Symmetry disabled: ", !basis.model.symmetries)

# Source position for delta function (fractional coordinates)
y = [0.5, 0.0, 0.0]  # Center of box

# Energy for Green's function
# For demonstration, use ground state energy from quick SCF
scfres = self_consistent_field(basis; tol=1e-4, maxiter=50)
E = scfres.energies.total

println("\nSystem properties:")
println("  Ground state energy: ", E)
println("  Source position y (fractional): ", y)

# Compute Green's function with full GMRES implementation
println("\nComputing Green's function...")

G = compute_periodic_green_function(basis, y, E;
                                   alpha=0.1,    # h(k) scaling
                                   deltaE=0.1,   # Energy width
                                   n_bands=5,    # Number of bands
                                   tol=1e-4,     # GMRES tolerance
                                   maxiter=50)   # GMRES max iterations

println("  Green's function computed!")
println("  Size: ", size(G))
println("  Type: ", typeof(G))
println("  Max |G|: ", maximum(abs, G))

# The Green's function G(x,y;E) should satisfy:
# (E - H) G(x,y;E) = δ(x-y) (modulo periodicity)

println("\nImplementation notes:")
println("  ✓ h(k) computation via Hellmann-Feynman theorem")
println("  ✓ Periodized delta function")
println("  ✓ Assembly framework with det(I+i∇h) weighting")
println("  ✓ Complex k-point Hamiltonian via kinetic correction")
println("  ✓ GMRES solver using KrylovKit")
println("\nFor full implementation details, see:")
println("  - src/postprocess/green_functions.jl")
println("  - test/green_functions.jl")
println("  - https://hal.science/hal-03611185/document")
