using PyCall
using DFTK

# Calculation parameters
kgrid = [3, 3, 3]     # Monkhorst-Pack k-Point sampling in each dimension
Ecut = 5              # Kinetic energy cutoff in Hartree
n_bands = 10          # Number of bands to compute
kline_density = 20    # Density of kpoints for band computation

# Setup silicon lattice
a = 10.263141334305942  # Silicon lattice constant in Bohr
lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
Si = Species(14)
composition = [Si => [ones(3)/8, -ones(3)/8]]

# Setup free-electron model with 8 electrons
n_electrons = 8
model = Model(lattice, n_electrons)

# Setup basis (with uniform k-Point mesh)
kpoints, ksymops = bzmesh_uniform(kgrid)
fft_size = determine_grid_size(lattice, Ecut)
basis = PlaneWaveModel(model, fft_size, Ecut, kpoints, ksymops)

# Free-electron band structure calculation
ham = Hamiltonian(basis)
kpoints, klabels, kpath = determine_high_symmetry_kpath(basis, kline_density, composition...)
println("Computing bands along kpath:\n     $(join(kpath, " -> "))")
band_data = compute_bands(ham, kpoints, n_bands)

# Plot bandstructure using pymatgen
plotter = pyimport("pymatgen.electronic_structure.plotter")
bs = pymatgen_bandstructure(basis, band_data, klabels)
bsplot = plotter.BSPlotter(bs)
plt = bsplot.get_plot()
plt.autoscale()
plt.savefig("silicon_free_electron.pdf")
plt.show()
