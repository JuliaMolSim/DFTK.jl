using DFTK
using LinearAlgebra
using CairoMakie

# Setup 2D system 
a = 1.0  # Box size in atomic units
lattice = a .* [[1 0 0.]; [0 1 0.]; [0 0 0]]  # 2D lattice

# Supercell range parameter: use -Rmax:Rmax for R_vectors
Rmax = 2  # integer; produces a (2*Rmax+1)×(2*Rmax+1) supercell

# Random-looking 2D periodic potential
# V(x,y) = V0 * [cos(2πx/a) + sin(2πy/a) + 0.5*cos(4πx/a)*sin(4πy/a)]
V0 = 1.0
alpha = 0.02
deltaE = 2
E = 5.  # Middle of first band
function potential(r)
    x, y = r[1], r[2]
    return V0 * (cos(2π * x / a) + cos(2π * y / a) + 0.5 * cos(4π * x / a) * sin(4π * y / a))
end

# Build model
n_electrons = 0  # 2D needs at least 2 electrons
terms = [
    Kinetic(),
    ExternalFromReal(r -> potential(r)),
]
model = Model(lattice; n_electrons, terms, spin_polarization=:spinless, symmetries=false)

# Create plane-wave basis
Ecut = 1000
nkpt = 10
kgrid = MonkhorstPack([nkpt, nkpt, 1])
basis = PlaneWaveBasis(model; Ecut, kgrid)

println("Basis setup:")
println("  Grid size: ", basis.fft_size)
println("  Number of k-points: ", length(basis.kpoints))

# Source position (fractional coordinates)
y = [0.0, 0.0, 0.0]

# Compute eigenvalues to find E in middle of first band
println("\nComputing eigenvalues...")
ham = Hamiltonian(basis)
n_bands = 5
eigres = diagonalize_all_kblocks(DFTK.lobpcg_hyper, ham, n_bands)

# Get all first-band eigenvalues
first_band_energies = [eigres.λ[ik][1] for ik in 1:length(basis.kpoints)]
E_min = minimum(first_band_energies)
E_max = maximum(first_band_energies)

println("First band energy range: [$(E_min), $(E_max)]")
println("Chosen E = $(E)")

# Compute Green's function with new API using Rmin/Rmax
println("\nComputing Green's function...")
Rmin = [-Rmax, -Rmax, 0]
Rmax_vec = [Rmax, Rmax, 0]
result = compute_periodic_green_function(basis, y, E;
                                        alpha=alpha, deltaE=deltaE, n_bands=n_bands,
                                        tol=1e-6, maxiter=100, 
                                        Rmin=Rmin, Rmax=Rmax_vec)

# Destructure returned values
G_dict = result.G_dict
G_extended_3d = result.G_extended
r_frac = result.r_frac_coords
s_frac = result.s_frac_coords

# Convert fractional coordinates to Cartesian
# For 2D system: x = r * a[1] + s * a[2] where a[1], a[2] are lattice vectors
lat = basis.model.lattice
x_extended = [r * lat[1, 1] + s * lat[1, 2] for r in r_frac, s in s_frac]
y_extended = [r * lat[2, 1] + s * lat[2, 2] for r in r_frac, s in s_frac]

println("  Green's function computed!")
println("  Number of cells: ", length(G_dict))

# ===== Prepare data for plotting =====

# Extract 2D slice from 3D extended array
G_extended = G_extended_3d[:, :, 1]

# For heatmap, we need 1D coordinate vectors (assuming grid is regular)
x_coords_1d = x_extended[:, 1]
y_coords_1d = y_extended[1, :]

# Extract k-point data for quiver plot
k_coords = [kpt.coordinate[1:2] for kpt in basis.kpoints]
k_x = [k[1] for k in k_coords]
k_y = [k[2] for k in k_coords]

# Compute h_values for visualization
h_values = DFTK.compute_h_values(basis, eigres, E, alpha, deltaE)
h_x = [h[1] for h in h_values]
h_y = [h[2] for h in h_values]

# Get first eigenvalue at each k-point for contour
eigenvalue_grid = zeros(length(k_x))
for ik in 1:length(basis.kpoints)
    eigenvalue_grid[ik] = eigres.λ[ik][1]
end

# ===== Create plots =====

fig = Figure(size=(1200, 600))

# Panel 1: Imaginary part of Green's function in (2*Rmax+1)×(2*Rmax+1) cells
ncell = 2 * Rmax + 1
ax1 = Axis(fig[1, 1], 
          xlabel="x (a.u.)", 
          ylabel="y (a.u.)",
          title="Im[G(x,y;E)] in $(ncell)×$(ncell) cells",
          aspect=DataAspect())

hm1 = heatmap!(ax1, x_coords_1d, y_coords_1d, imag.(G_extended), 
              colormap=:RdBu, colorrange=(-maximum(abs, imag.(G_extended)), 
                                          maximum(abs, imag.(G_extended))), interpolate=true)
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

# Heatmap of eigenvalues (with interpolation)
hm2 = heatmap!(ax2, k_unique_x, k_unique_y, eigenvalue_matrix', colormap=:viridis, interpolate=true)
Colorbar(fig[1, 4], hm2, label="λ₁(k)")

# Contour at energy E
contour!(ax2, k_unique_x, k_unique_y, eigenvalue_matrix', 
        levels=[E], color=:red, linewidth=2, label="E contour")

# Quiver plot for h(k)
# Scale arrows for visibility
scale_factor = 0.5# / maximum(sqrt.(real.(h_x).^2 + real.(h_y).^2))
arrows2d!(ax2, k_x, k_y, 
          scale_factor .* real.(h_x), scale_factor .* real.(h_y),
          lengthscale=1.0, color=:white)

# Add legend
Legend(fig[2, 1:4], ax2, "Legend", orientation=:horizontal)

println("\nDisplaying plot...")
display(fig)

println("\nPlot complete!")
println("Panel 1: Imaginary part of G(x,y;E) over $(ncell)×$(ncell) unit cells")
println("Panel 2: First eigenvalue λ₁(k), contour at E=$(round(E, digits=3)), and h(k) quiver")
