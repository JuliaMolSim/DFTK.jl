using DFTK
using Printf
using PyCall

# Calculation parameters
calculation_model = :lda
kgrid = [4, 4, 4]        # k-Point grid
supercell = [1, 1, 1]    # Lattice supercell
Ecut = 15                # kinetic energy cutoff in Hartree
n_bands_plot = 8 # number of bands to plot in the bandstructure
kline_density = 20       # Density of k-Points for bandstructure

# Setup silicon lattice
a = 10.263141334305942  # Silicon lattice constant in Bohr
lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
Si = Species(14, psp=load_psp("si-pade-q4.hgh"))
composition = [Si => [ones(3)/8, -ones(3)/8]]

# Make a supercell if desired
pystruct = pymatgen_structure(lattice, composition...)
pystruct.make_supercell(supercell)
for i in 1:3, j in 1:3
    A_to_bohr = pyimport("pymatgen.core.units").ang_to_bohr
    lattice[i, j] = A_to_bohr * get(get(pystruct.lattice.matrix, j-1), i-1)
end
composition = [Si => [s.frac_coords for s in pystruct.sites]]

model = nothing
# Setup model and discretisation
if calculation_model == :reduced_hf
    model = model_reduced_hf(lattice, composition...)
elseif calculation_model == :lda
    model = model_dft(lattice, :lda_xc_teter93, composition...)
elseif calculation_model == :pbe
    model = model_dft(lattice, [:gga_x_pbe, :gga_c_pbe], composition...)
elseif calculation_model == :indep_electrons
    model = model_hcore(lattice, composition...)
elseif calculation_model == :free
    n_electrons = 8*prod(supercell)
    model = Model(lattice, n_electrons)
else
    error("Unknown calculation_model $(calculation_model)")
end
kcoords, ksymops = bzmesh_ir_wedge(kgrid, lattice, composition...)
fft_size = determine_grid_size(lattice, Ecut)
basis = PlaneWaveModel(model, fft_size, Ecut, kcoords, ksymops)

# Run SCF, note Silicon metal is an insulator, so we will assume that we do not need
# all bands here. This will cause warnings in some models, because e.g. in the
# :reduced_hf model silicon is a metal
n_bands_scf = Int(model.n_electrons / 2)
ham = Hamiltonian(basis, guess_gaussian_sad(basis, composition...))
scfres = self_consistent_field!(ham, n_bands_scf, tol=1e-6)

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
println("Computing bands along kpath:\n     $(join(kpath[1], " -> "))")
band_data = compute_bands(ham, kpoints, n_bands_plot)

# Plot bandstructure using pymatgen
plotter = pyimport("pymatgen.electronic_structure.plotter")
bs = pymatgen_bandstructure(basis, band_data, klabels, fermi_level=scfres.εF)
bsplot = plotter.BSPlotter(bs)
plt = bsplot.get_plot()
plt.autoscale()
plt.savefig("silicon_$(calculation_model).pdf")
plt.show()
