# Example: Computing Periodic Green's Functions in 1D
# 
# This example demonstrates the periodic Green's function computation 
# framework for a 1D harmonic oscillator system.

using DFTK
using LinearAlgebra

# Setup 1D system 
a = 1.0  # Box size in atomic units
lattice = a .* [[1 0 0.]; [0 0 0]; [0 0 0]]  # 1D lattice

# Periodic cosine potential V(x) = cos(2π x / a)
potential(x) = 1*cos(2π * x / a)

E = 2

# Build model (simple kinetic + external potential, spinless)
# Disable symmetry at model level for Green's function computation
n_electrons = 1
terms = [
    Kinetic(),
    ExternalFromReal(r -> potential(r[1])),
]
model = Model(lattice; n_electrons, terms, spin_polarization=:spinless, symmetries=false)

# Create plane-wave basis
Ecut = 20000
kgrid = MonkhorstPack([4, 1, 1])  # 4 k-points in 1D
basis = PlaneWaveBasis(model; Ecut, kgrid)

# Compute and plot band structure
println("\nComputing band structure...")
 # = 50
bands_kpath = compute_bands(basis; kline_density=40, n_bands=3)

using Plots
p = plot_bandstructure(bands_kpath; ylims=(-2, 4))
title!(p, "1D Band Structure (Periodic Cosine Potential)")
xlabel!(p, "k-point")
ylabel!(p, "Energy (Ha)")
display(p)
println("Band structure plotted!")





# Source position for delta function (fractional coordinates)
y = [0.5, 0.0, 0.0]  # Center of box

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
