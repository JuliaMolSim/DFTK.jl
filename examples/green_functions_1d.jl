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

using Plots

# ===== Panel 1: Band structure =====
println("\nComputing band structure...")
bands_kpath = compute_bands(basis; kline_density=40, n_bands=3)

p1 = plot_bandstructure(bands_kpath; ylims=(-2, 20))
title!(p1, "Band Structure")
xlabel!(p1, "k-point")
ylabel!(p1, "Energy (Ha)")
hline!(p1, [E], label="E = $E", linestyle=:dash, color=:black, linewidth=2)

# ===== Panel 2: h(k) and ∇h(k) =====
println("Computing h(k) and ∇h(k)...")
ham = DFTK.Hamiltonian(basis)
eigres = DFTK.diagonalize_all_kblocks(DFTK.lobpcg_hyper, ham, 5)
h_values = DFTK.compute_h_values(basis, eigres, E, alpha, deltaE)
nabla_h_values = DFTK.compute_nabla_h_finite_diff(basis, h_values)

# Extract k-coordinates (1D system, so only k_x is relevant)
k_coords = [kpt.coordinate[1] for kpt in basis.kpoints]
h_x = [h[1] for h in h_values]
nabla_h_xx = [nabla_h[1, 1] for nabla_h in nabla_h_values]

# Sort by k for plotting
sorted_indices = sortperm(k_coords)
k_sorted = k_coords[sorted_indices]
h_x_sorted = h_x[sorted_indices]
nabla_h_xx_sorted = nabla_h_xx[sorted_indices]

p2 = plot(k_sorted, real(h_x_sorted), label="h_x(k)", linewidth=2, marker=:circle,
          xlabel="k (fractional)", ylabel="Value", title="h(k) and derivatives")
plot!(p2, k_sorted, real(nabla_h_xx_sorted), label="∂h_x/∂k_x", 
      linewidth=2, marker=:square, linestyle=:dash, color=:red)

# ===== Panel 3: Green's function =====
println("Computing Green's function...")
G_dict = compute_periodic_green_function(basis, y, E;
                                        alpha=alpha, deltaE=deltaE, n_bands=5,
                                        tol=1e-8, maxiter=50, R_vectors=R_vectors)

println("  Green's function computed!")
println("  Number of cells: ", length(G_dict))
println("  Max |G|: ", maximum(maximum(abs, G) for G in values(G_dict)))

# Build extended x coordinates and G values
n_points_per_cell = size(first(values(G_dict)))[1]
x_all = Float64[]
G_all = ComplexF64[]

for R in R_vectors
    G_R = G_dict[R][:, 1, 1]
    x_cell = range(0, a, length=n_points_per_cell+1)[1:end-1]
    x_shifted = x_cell .+ R[1] * a
    append!(x_all, x_shifted)
    append!(G_all, G_R)
end

p3 = plot(x_all, real.(G_all), label="Re[G(x,y;E)]", linewidth=2,
          xlabel="x (a.u.)", ylabel="Green's function", title="Green's Function")
plot!(p3, x_all, imag.(G_all), label="Im[G(x,y;E)]", linewidth=2, linestyle=:dash)
vline!(p3, [y[1] * a], label="source", linestyle=:dot, linewidth=1, color=:red, alpha=0.5)

# ===== Combined plot =====
p_combined = plot(p1, p2, p3, layout=(3, 1), size=(800, 900))
display(p_combined)
println("All plots displayed!")
