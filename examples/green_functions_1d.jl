# Example: Computing Periodic Green's Functions in 1D
# 
# This example demonstrates the periodic Green's function computation 
# framework for a 1D harmonic oscillator system.

using DFTK
using LinearAlgebra

# Setup 1D system 
a = 1.0  # Box size in atomic units

# Source position for delta function (fractional coordinates)
y = [0.0, 0.0, 0.0]  # Center of box
E = -1
nkpoints = 50
Ecut = 200000
V0 = 0
alpha = 0.0im
deltaE = 5

# Define lattice vectors for extended range (plot over 5 unit cells centered at origin)
Rmax = 5
R_vectors = [[r, 0.0, 0.0] for r in -Rmax:Rmax]


lattice = a .* [[1 0 0.]; [0 0 0]; [0 0 0]]  # 1D lattice

# Periodic cosine potential V(x) = cos(2π x / a)
potential(x) = V0*cos(2π * x / a)

# Build model (simple kinetic + external potential, spinless)
# Disable symmetry at model level for Green's function computation
n_electrons = 1
terms = [
    Kinetic(),
    ExternalFromReal(r -> potential(r[1])),
]
model = Model(lattice; n_electrons, terms, spin_polarization=:spinless, symmetries=false)

# Create plane-wave basis
kgrid = MonkhorstPack([nkpoints, 1, 1])  # 4 k-points in 1D
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
# stop

# Compute h(k) and ∇h(k) for visualization
println("\nComputing h(k) and ∇h(k)...")
ham = DFTK.Hamiltonian(basis)
eigres = DFTK.diagonalize_all_kblocks(DFTK.lobpcg_hyper, ham, 5)
h_values = DFTK.compute_h_values(basis, eigres, E, alpha, deltaE)
nabla_h_values = DFTK.compute_nabla_h_finite_diff(basis, h_values)

# Extract k-coordinates (1D system, so only k_x is relevant)
k_coords = [kpt.coordinate[1] for kpt in basis.kpoints]
h_x = [h[1] for h in h_values]  # x-component of h(k)
nabla_h_xx = [nabla_h[1, 1] for nabla_h in nabla_h_values]  # ∂h_x/∂k_x

# Sort by k for plotting
sorted_indices = sortperm(k_coords)
k_sorted = k_coords[sorted_indices]
h_x_sorted = h_x[sorted_indices]
nabla_h_xx_sorted = nabla_h_xx[sorted_indices]

# Compute numerical derivative manually for verification
dh_dk_numerical = zeros(typeof(alpha), length(k_sorted))
for i in 2:length(k_sorted)-1
    dk = k_sorted[i+1] - k_sorted[i-1]
    dh_dk_numerical[i] = (h_x_sorted[i+1] - h_x_sorted[i-1]) / dk
end
# Use forward/backward difference at boundaries
dh_dk_numerical[1] = (h_x_sorted[2] - h_x_sorted[1]) / (k_sorted[2] - k_sorted[1])
dh_dk_numerical[end] = (h_x_sorted[end] - h_x_sorted[end-1]) / (k_sorted[end] - k_sorted[end-1])

# Plot h(k) and both derivatives for comparison
p_h = plot(k_sorted, real(h_x_sorted), label="h_x(k)", linewidth=2, marker=:circle,
           xlabel="k (fractional)", ylabel="Value", 
           title="h(k) and its derivatives", legend=:best)
plot!(p_h, k_sorted, real(nabla_h_xx_sorted), label="∂h_x/∂k_x (finite diff)", 
      linewidth=2, marker=:square, linestyle=:dash, color=:red)
plot!(p_h, k_sorted, real(dh_dk_numerical), label="∂h_x/∂k_x (numerical)", 
      linewidth=2, marker=:diamond, color=:green)
display(p_h)
println("h(k) and ∇h(k) plotted!")
# stop


G_dict = compute_periodic_green_function(basis, y, E;
                                        alpha=alpha,    # h(k) scaling
                                        deltaE=deltaE,   # Energy width
                                        n_bands=5,    # Number of bands
                                        tol=1e-8,     # GMRES tolerance
                                        maxiter=50,   # GMRES max iterations
                                        R_vectors=R_vectors)

println("  Green's function computed!")
println("  Number of cells: ", length(G_dict))
println("  Size per cell: ", size(first(values(G_dict))))
println("  Max |G| overall: ", maximum(maximum(abs, G) for G in values(G_dict)))

# The Green's function G(x,y;E) should satisfy:
# (E - H) G(x,y;E) = δ(x-y) (modulo periodicity)

# Plot the Green's function G(x,y;E) as a function of x across multiple cells
println("\nPlotting Green's function...")

# Build extended x coordinates and G values
n_points_per_cell = size(first(values(G_dict)))[1]
x_all = Float64[]
G_all = ComplexF64[]

for R in R_vectors
    G_R = G_dict[R][:, 1, 1]  # Extract 1D slice
    x_cell = range(0, a, length=n_points_per_cell+1)[1:end-1]
    # Shift by lattice vector
    x_shifted = x_cell .+ R[1] * a
    append!(x_all, x_shifted)
    append!(G_all, G_R)
end

# Plot real and imaginary parts across multiple cells
p_green = plot(x_all, real.(G_all), label="Re[G(x,y;E)]", 
               linewidth=2, xlabel="x (a.u.)", ylabel="Green's function",
               title="Green's Function G(x,y;E) - Multiple Unit Cells")
plot!(p_green, x_all, imag.(G_all), label="Im[G(x,y;E)]", 
      linewidth=2, linestyle=:dash)
# Mark source position (periodically repeated)
for R in R_vectors
    vline!(p_green, [(y[1] + R[1]) * a], label=(R == R_vectors[1] ? "Source y+R" : ""), 
           linestyle=:dot, linewidth=1, color=:red, alpha=0.5)
end
display(p_green)
println("Green's function plotted!")
