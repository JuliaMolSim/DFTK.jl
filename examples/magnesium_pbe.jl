using PyCall
using DFTK
using Printf

# Calculation parameters
kgrid = [4, 4, 4]        # k-Point grid
Ecut = 15                # kinetic energy cutoff in Hartree
supercell = [1, 1, 1]    # Lattice supercell
n_bands = 8              # Number of bands for SCF and plotting
Tsmear = 0.01            # Smearing temperature in Hartree
kline_density = 20       # Density of k-Points for bandstructure

# Setup manganese lattice (constants in Bohr)
a = 3.0179389193174084
b = 5.227223542397263
c = 9.773621942589742
lattice = [[-a -b  0]; [-a  b  0]; [0   0 -c]]
Mg = Species(12, psp=load_psp("mg-pbe-q2.hgh"))
composition = [Mg => [[2/3, 1/3, 1/4], [1/3, 2/3, 3/4]]]

# Make a supercell if desired
pystruct = pymatgen_structure(lattice, composition...)
pystruct.make_supercell(supercell)
for i in 1:3, j in 1:3
    A_to_bohr = pyimport("pymatgen.core.units").ang_to_bohr
    lattice[i, j] = A_to_bohr * get(get(pystruct.lattice.matrix, j-1), i-1)
end
composition = [Mg => [s.frac_coords for s in pystruct.sites]]

# Setup PBE model with Methfessel-Paxton smearing and its discretisation
model = model_dft(lattice, [:gga_x_pbe, :gga_c_pbe], composition...;
                  temperature=Tsmear,
                  smearing=DFTK.smearing_methfessel_paxton_1)
kcoords, ksymops = bzmesh_ir_wedge(kgrid, lattice, composition...)
basis = PlaneWaveBasis(model, Ecut, kcoords, ksymops)

# Run SCF
ham = Hamiltonian(basis, guess_density(basis, composition...))
scfres = self_consistent_field(ham, n_bands)
ham = scfres.ham

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
kpoints, klabels, kpath = high_symmetry_kpath(basis, kline_density, composition...)
println("Computing bands along kpath:\n     $(join(kpath[1], " -> "))")
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
