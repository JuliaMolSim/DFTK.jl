using PyCall
using DFTK
using Printf

# Calculation parameters
kgrid = [4, 4, 4]
Ecut = 15  # Hartree
n_bands = 8
Tsmear = 0.01
kline_density = 20

# Setup manganese lattice (constants in Bohr)
a = 3.0179389193174084
b = 5.227223542397263
c = 9.773621942589742
lattice = [[-a -b  0]; [-a  b  0]; [0   0 -c]]
Mg = Species(12, psp=load_psp("mg-pbe-q2.hgh"))
composition = [Si => [[2/3, 1/3, 1/4], [1/3, 2/3, 3/4]]]

# Setup PBE model with Methfessel-Paxton smearing and its discretisation
model = model_dft(lattice, [:gga_x_pbe, :gga_c_pbe], composition...;
                  temperature=Tsmear,
                  smearing=DFTK.smearing_methfessel_paxton_1)
kpoints, ksymops = bzmesh_ir_wedge(kgrid, lattice, composition...)
fft_size = determine_grid_size(lattice, Ecut)
basis = PlaneWaveModel(model, fft_size, Ecut, kpoints, ksymops)

# Run SCF
ham = Hamiltonian(basis, guess_gaussian_sad(basis, composition...))
scfres = self_consistent_field!(ham, n_bands)

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
plt.savefig("magnesium_pbe.pdf")
plt.legend()
plt.show()
