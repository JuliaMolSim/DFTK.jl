using PyCall
using DFTK
using Printf

# Calculation parameters
kgrid = [4, 4, 4]
Ecut = 25  # Hartree
n_bands = 8
kline_density = 20

# Setup silicon lattice and structure
a = 10.263141334305942  # Silicon lattice constant in Bohr
lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
Si = Species(14, psp=load_psp("si-pbe-q4.hgh"))
composition = [Si => [ones(3)/8, -ones(3)/8]]

# Setup PBE model and discretisation
model = model_dft(lattice, [:gga_x_pbe, :gga_c_pbe], composition...)
kpoints, ksymops = bzmesh_ir_wedge(kgrid, lattice, composition...)
fft_size = determine_grid_size(lattice, Ecut)
basis = PlaneWaveModel(model, fft_size, Ecut, kpoints, ksymops)

# Run SCF, note Silicon metal is an insulator, so no need for all bands here
n_bands_scf = Int(model.n_electrons / 2)
ham = Hamiltonian(basis, guess_gaussian_sad(basis, composition...))
scfres = self_consistent_field!(ham, n_bands_scf, tol=1e-8)

# Print obtained energies
energies = scfres.energies
energies[:Ewald] = energy_nuclear_ewald(model.lattice, composition...)
energies[:PspCorrection] = energy_nuclear_psp_correction(model.lattice, composition...)
println("\nEnergy breakdown:")
for key in sort([keys(energies)...]; by=S -> string(S))
    @printf "    %-20s%-10.7f\n" string(key) energies[key]
end
@printf "\n    %-20s%-15.12f\n\n" "total" sum(values(energies))

# Band structure calculation along high-symmetry path
kpoints, klabels, kpath = determine_high_symmetry_kpath(basis, kline_density, composition...)
println("Computing bands along kpath:\n     $(join(kpath, " -> "))")
band_data = compute_bands(ham, kpoints, n_bands)

# Plot bandstructure using pymatgen
plotter = pyimport("pymatgen.electronic_structure.plotter")
bs = pymatgen_bandstructure(basis, band_data, klabels, fermi_level=scfres.εF)
bsplot = plotter.BSPlotter(bs)
plt = bsplot.get_plot()
plt.autoscale()
plt.savefig("silicon_pbe.pdf")
plt.legend()
plt.show()
