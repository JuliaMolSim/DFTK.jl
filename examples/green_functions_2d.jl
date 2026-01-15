using DFTK
using LinearAlgebra
using CairoMakie

# Setup 2D system 
a = 10.0  # Box size in atomic units
lattice = a .* [[1 0 0.]; [0 1 0.]; [0 0 0]]  # 2D lattice

# Random-looking 2D periodic potential
# V(x,y) = V0 * [cos(2πx/a) + sin(2πy/a) + 0.5*cos(4πx/a)*sin(4πy/a)]
V0 = 1.0
function potential(r)
    x, y = r[1], r[2]
    return V0 * (cos(2π * x / a) + sin(2π * y / a) + 0.5 * cos(4π * x / a) * sin(4π * y / a))
end

# Build model
n_electrons = 2  # 2D needs at least 2 electrons
terms = [
    Kinetic(),
    ExternalFromReal(r -> potential(r)),
]
model = Model(lattice; n_electrons, terms, spin_polarization=:spinless, symmetries=false)

# Create plane-wave basis
Ecut = 50
kgrid = MonkhorstPack([4, 4, 1])  # 4×4 k-points in 2D
basis = PlaneWaveBasis(model; Ecut, kgrid)

println("Basis setup:")
println("  Grid size: ", basis.fft_size)
println("  Number of k-points: ", length(basis.kpoints))

# Source position (fractional coordinates)
y = [0.5, 0.5, 0.0]

# Compute eigenvalues to find E in middle of first band
println("\nComputing eigenvalues...")
ham = Hamiltonian(basis)
n_bands = 5
eigres = diagonalize_all_kblocks(diag_full, ham, n_bands)

# Get all first-band eigenvalues
first_band_energies = [eigres.λ[ik][1] for ik in 1:length(basis.kpoints)]
E_min = minimum(first_band_energies)
E_max = maximum(first_band_energies)
E = (E_min + E_max) / 2  # Middle of first band

println("First band energy range: [$(E_min), $(E_max)]")
println("Chosen E = $(E)")

# Compute h(k) and ∇h(k)
println("\nComputing h(k) and ∇h(k)...")
alpha = 0.1
deltaE = 0.1
h_values = compute_h_values(basis, eigres, E, alpha, deltaE)
nabla_h_values = compute_nabla_h_finite_diff(basis, h_values)

# Define lattice vectors for 3×3 cells
R_vectors = [[i, j, 0.0] for i in -1:1 for j in -1:1]

# Compute Green's function
println("Computing Green's function...")
G_dict = compute_periodic_green_function(basis, y, E;
                                        alpha=alpha, deltaE=deltaE, n_bands=n_bands,
                                        tol=1e-6, maxiter=100, R_vectors=R_vectors)

println("  Green's function computed!")
println("  Number of cells: ", length(G_dict))

# ===== Prepare data for plotting =====

# Build extended Green's function on 3×3 grid
nx, ny, nz = basis.fft_size
G_extended = zeros(ComplexF64, 3*nx, 3*ny)

for (iR, R) in enumerate(R_vectors)
    i, j = Int(R[1]), Int(R[2])
    G_R = G_dict[R][:, :, 1]  # Extract 2D slice
    
    # Place in extended grid
    ix_start = (i + 1) * nx + 1
    iy_start = (j + 1) * ny + 1
    G_extended[ix_start:ix_start+nx-1, iy_start:iy_start+ny-1] = G_R
end

# Create coordinate arrays for extended grid
x_extended = range(-a, 2*a, length=3*nx)
y_extended = range(-a, 2*a, length=3*ny)

# Extract k-point data for quiver plot
k_coords = [kpt.coordinate[1:2] for kpt in basis.kpoints]
k_x = [k[1] for k in k_coords]
k_y = [k[2] for k in k_coords]
h_x = [h[1] for h in h_values]
h_y = [h[2] for h in h_values]

# Get first eigenvalue at each k-point for contour
eigenvalue_grid = zeros(length(k_x))
for ik in 1:length(basis.kpoints)
    eigenvalue_grid[ik] = eigres.λ[ik][1]
end

# ===== Create plots =====

fig = Figure(size=(1200, 600))

# Panel 1: Imaginary part of Green's function in 3×3 cells
ax1 = Axis(fig[1, 1], 
          xlabel="x (a.u.)", 
          ylabel="y (a.u.)",
          title="Im[G(x,y;E)] in 3×3 cells",
          aspect=DataAspect())

hm1 = heatmap!(ax1, x_extended, y_extended, imag.(G_extended), 
              colormap=:RdBu, colorrange=(-maximum(abs, imag.(G_extended)), 
                                          maximum(abs, imag.(G_extended))))
Colorbar(fig[1, 2], hm1, label="Im[G]")

# Mark source position
scatter!(ax1, [y[1]*a], [y[2]*a], color=:green, markersize=15, marker=:star5, label="source")

# Panel 2: First eigenvalue heatmap with contour at E and h(k) quiver
ax2 = Axis(fig[1, 3], 
          xlabel="kₓ (fractional)", 
          ylabel="kᵧ (fractional)",
          title="First band λ(k), contour at E, and h(k)",
          aspect=DataAspect())

# Create regular grid for interpolation
k_unique_x = sort(unique(k_x))
k_unique_y = sort(unique(k_y))
eigenvalue_matrix = zeros(length(k_unique_x), length(k_unique_y))

for ik in 1:length(basis.kpoints)
    ix = findfirst(==(k_x[ik]), k_unique_x)
    iy = findfirst(==(k_y[ik]), k_unique_y)
    eigenvalue_matrix[ix, iy] = eigenvalue_grid[ik]
end

# Heatmap of eigenvalues
hm2 = heatmap!(ax2, k_unique_x, k_unique_y, eigenvalue_matrix', colormap=:viridis)
Colorbar(fig[1, 4], hm2, label="λ₁(k)")

# Contour at energy E
contour!(ax2, k_unique_x, k_unique_y, eigenvalue_matrix', 
        levels=[E], color=:red, linewidth=2, label="E contour")

# Quiver plot for h(k)
# Scale arrows for visibility
scale_factor = 0.02 / maximum(sqrt.(real.(h_x).^2 + real.(h_y).^2))
arrows!(ax2, k_x, k_y, 
       scale_factor .* real.(h_x), scale_factor .* real.(h_y),
       arrowsize=10, lengthscale=1.0, color=:white, linewidth=1.5)

# Add legend
Legend(fig[2, 1:4], ax2, "Legend", orientation=:horizontal)

println("\nDisplaying plot...")
display(fig)

println("\nPlot complete!")
println("Panel 1: Imaginary part of G(x,y;E) over 3×3 unit cells")
println("Panel 2: First eigenvalue λ₁(k), contour at E=$(round(E, digits=3)), and h(k) quiver")
