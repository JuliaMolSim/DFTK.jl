using DFTK
using LinearAlgebra
using WGLMakie

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
E = 5.0
function potential(r)
    x, y = r[1], r[2]
    return V0 * (cos(2π * x / a) + cos(2π * y / a) + 0.5 * cos(4π * x / a) * sin(4π * y / a))
end

# Build model
n_electrons = 0
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

# Initial E for single frame plot
println("Initial E for plot = $(E)")

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

# Get eigenvalues for first three bands
eigenvalue_grid_1 = zeros(length(k_x))
eigenvalue_grid_2 = zeros(length(k_x))
eigenvalue_grid_3 = zeros(length(k_x))
for ik in 1:length(basis.kpoints)
    eigenvalue_grid_1[ik] = eigres.λ[ik][1]
    eigenvalue_grid_2[ik] = eigres.λ[ik][2]
    eigenvalue_grid_3[ik] = eigres.λ[ik][3]
end

# ===== Create plots =====

fig = Figure(size=(1800, 600))

# Panel 1: 3D band structure (first three bands)
k_unique_x = sort(unique(k_x))
k_unique_y = sort(unique(k_y))

# Create matrices for three bands
E1_matrix = zeros(length(k_unique_x), length(k_unique_y))
E2_matrix = zeros(length(k_unique_x), length(k_unique_y))
E3_matrix = zeros(length(k_unique_x), length(k_unique_y))

for ik in 1:length(basis.kpoints)
    ix = findfirst(==(k_x[ik]), k_unique_x)
    iy = findfirst(==(k_y[ik]), k_unique_y)
    E1_matrix[ix, iy] = eigenvalue_grid_1[ik]
    E2_matrix[ix, iy] = eigenvalue_grid_2[ik]
    E3_matrix[ix, iy] = eigenvalue_grid_3[ik]
end

ax1 = Axis3(fig[1, 1], xlabel="kₓ", ylabel="kᵧ", zlabel="Energy (Ha)",
            title="First Three Bands", azimuth=0.8π, elevation=0.03π)

# Plot three bands as surfaces
surface!(ax1, k_unique_x, k_unique_y, E1_matrix', color=:red, alpha=0.7)
surface!(ax1, k_unique_x, k_unique_y, E2_matrix', color=:blue, alpha=0.7)
surface!(ax1, k_unique_x, k_unique_y, E3_matrix', color=:green, alpha=0.7)

# Add horizontal plane at E
xlims = extrema(k_unique_x)
ylims = extrema(k_unique_y)
E_plane = fill(E, 2, 2)
surface!(ax1, [xlims[1], xlims[2]], [ylims[1], ylims[2]], E_plane, 
         color=:black, alpha=0.3)

# Panel 2: Real part of Green's function
ax2 = Axis(fig[1, 2], 
          xlabel="x (a.u.)", 
          ylabel="y (a.u.)",
          title="Re[G(x,y;E)]",
          aspect=DataAspect())

hm2 = heatmap!(ax2, x_coords_1d, y_coords_1d, real.(G_extended), 
              colormap=:RdBu, interpolate=true)
Colorbar(fig[1, 3], hm2, label="Re[G]")
scatter!(ax2, [y[1]*a], [y[2]*a], color=:green, markersize=15, marker=:star5)

# Panel 3: Imaginary part of Green's function
ax3 = Axis(fig[1, 4], 
          xlabel="x (a.u.)", 
          ylabel="y (a.u.)",
          title="Im[G(x,y;E)]",
          aspect=DataAspect())

hm3 = heatmap!(ax3, x_coords_1d, y_coords_1d, imag.(G_extended), 
              colormap=:RdBu, interpolate=true)
Colorbar(fig[1, 5], hm3, label="Im[G]")
scatter!(ax3, [y[1]*a], [y[2]*a], color=:green, markersize=15, marker=:star5)

println("\nDisplaying plot...")
display(fig)

println("\nPlot complete!")
println("Panel 1: First three bands in 3D with plane at E=$(round(E, digits=3))")
println("Panel 2: Real part of G(x,y;E)")
println("Panel 3: Imaginary part of G(x,y;E)")

# ===== Create video sweeping E from -10 to 20 =====
println("\n" * "="^60)
println("Creating video with E sweep from -10 to 20...")
println("="^60)

# Video parameters
nframes = 400
duration = 300 #seconds
E_values = range(-10, 20, length=nframes)
fps = nframes/duration  # frames per second for 10s duration
output_file = "green_function_E_sweep.mp4"

# Pre-compute global color ranges by sampling a few energies
println("Pre-computing color ranges...")
sample_energies = [E_values[1], E_values[end÷2], E_values[end]]
G_samples = []
for E_sample in sample_energies
    result_sample = compute_periodic_green_function(basis, y, E_sample;
                                                   alpha=alpha, deltaE=deltaE, n_bands=n_bands,
                                                   tol=1e-6, maxiter=100,
                                                   Rmin=Rmin, Rmax=Rmax_vec)
    push!(G_samples, result_sample.G_extended[:, :, 1])
end

# Compute global color ranges
all_real_vals = vcat([vec(real.(G)) for G in G_samples]...)
all_imag_vals = vcat([vec(imag.(G)) for G in G_samples]...)
colorrange_re = (minimum(all_real_vals), maximum(all_real_vals))
colorrange_im = (minimum(all_imag_vals), maximum(all_imag_vals))

println("  Re[G] range: $colorrange_re")
println("  Im[G] range: $colorrange_im")

# Create figure for video
fig_video = Figure(size=(1800, 600))

# Setup axes (reuse structure)
ax1_v = Axis3(fig_video[1, 1], xlabel="kₓ", ylabel="kᵧ", zlabel="Energy (Ha)",
              title="First Three Bands", azimuth=0.8π, elevation=0.03π)
ax2_v = Axis(fig_video[1, 2], xlabel="x (a.u.)", ylabel="y (a.u.)",
             title="Re[G(x,y;E)]", aspect=DataAspect())
ax3_v = Axis(fig_video[1, 4], xlabel="x (a.u.)", ylabel="y (a.u.)",
             title="Im[G(x,y;E)]", aspect=DataAspect())

# Create colorbars once with fixed ranges
cb2 = Colorbar(fig_video[1, 3], colormap=:RdBu, colorrange=colorrange_re, label="Re[G]")
cb3 = Colorbar(fig_video[1, 5], colormap=:RdBu, colorrange=colorrange_im, label="Im[G]")

# Start recording
record(fig_video, output_file, enumerate(E_values); framerate=fps) do (iframe, E_current)
    println("Frame $iframe/$(length(E_values)): E = $(round(E_current, digits=3))")
    
    # Clear axes
    empty!(ax1_v)
    empty!(ax2_v)
    empty!(ax3_v)
    
    # Recompute h_values and Green's function for this E
    h_vals = DFTK.compute_h_values(basis, eigres, E_current, alpha, deltaE)
    
    # Compute Green's function
    result_video = compute_periodic_green_function(basis, y, E_current;
                                                   alpha=alpha, deltaE=deltaE, n_bands=n_bands,
                                                   tol=1e-6, maxiter=100,
                                                   Rmin=Rmin, Rmax=Rmax_vec)
    G_ext_video = result_video.G_extended[:, :, 1]
    

    # Plot three bands as surfaces
    surface!(ax1_v, k_unique_x, k_unique_y, E1_matrix', color=:red, alpha=0.7)
    surface!(ax1_v, k_unique_x, k_unique_y, E2_matrix', color=:blue, alpha=0.7)
    surface!(ax1_v, k_unique_x, k_unique_y, E3_matrix', color=:green, alpha=0.7)

    # Add horizontal plane at E
    xlims = extrema(k_unique_x)
    ylims = extrema(k_unique_y)
    E_plane = fill(E_current, 2, 2)
    surface!(ax1_v, [xlims[1], xlims[2]], [ylims[1], ylims[2]], E_plane, 
             color=:black, alpha=0.3)

    
    
    # Update Re[G] with fixed colorrange
    heatmap!(ax2_v, x_coords_1d, y_coords_1d, real.(G_ext_video),
             colormap=:RdBu, colorrange=colorrange_re, interpolate=true)
    scatter!(ax2_v, [y[1]*a], [y[2]*a], color=:green, markersize=15, marker=:star5)
    
    # Update Im[G] with fixed colorrange
    heatmap!(ax3_v, x_coords_1d, y_coords_1d, imag.(G_ext_video),
             colormap=:RdBu, colorrange=colorrange_im, interpolate=true)
    scatter!(ax3_v, [y[1]*a], [y[2]*a], color=:green, markersize=15, marker=:star5)
    
    # Update titles with current E
    ax1_v.title = "First Three Bands (E = $(round(E_current, digits=2)))"
    ax2_v.title = "Re[G(x,y;E = $(round(E_current, digits=2)))]"
    ax3_v.title = "Im[G(x,y;E = $(round(E_current, digits=2)))]"
end

println("\nVideo saved to: $output_file")
println("Duration: 10 seconds, $(length(E_values)) frames at $fps fps")
