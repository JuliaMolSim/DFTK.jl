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
Si = Species(14, psp=load_psp("hgh/lda/Si-q4"))
# Si = Species(14)
atoms = [Si => [ones(3)/8, -ones(3)/8]]

# Make a supercell if desired
pystruct = pymatgen_structure(lattice, atoms)
pystruct.make_supercell(supercell)
lattice = load_lattice(pystruct)
atoms = [Si => [s.frac_coords for s in pystruct.sites]]

model = nothing
# Setup model and discretisation
if calculation_model == :reduced_hf
    model = model_reduced_hf(lattice, atoms)
elseif calculation_model == :lda
    model = model_dft(lattice, :lda_xc_teter93, atoms)
elseif calculation_model == :pbe
    model = model_dft(lattice, [:gga_x_pbe, :gga_c_pbe], atoms)
elseif calculation_model == :indep_electrons
    model = model_hcore(lattice, atoms)
elseif calculation_model == :free
    n_electrons = 8*prod(supercell)
    model = Model(lattice, n_electrons)
else
    error("Unknown calculation_model $(calculation_model)")
end
kcoords, ksymops = bzmesh_ir_wedge(kgrid, lattice, atoms)
basis = PlaneWaveBasis(model, Ecut, kcoords, ksymops)

# Run SCF. Note Silicon is a semiconductor, so we use an insulator
# occupation scheme. This will cause warnings in some models, because
# e.g. in the :reduced_hf model silicon is a metal
n_bands_scf = Int(model.n_electrons / 2)
ham = Hamiltonian(basis, guess_density(basis))
scfres = self_consistent_field(ham, n_bands_scf, tol=1e-6)
ham = scfres.ham

# Print obtained energies
energies = scfres.energies
energies[:Ewald] = energy_nuclear_ewald(model)
energies[:PspCorrection] = energy_nuclear_psp_correction(model)
println("\nEnergy breakdown:")
for key in sort([keys(energies)...]; by=S -> string(S))
    @printf "    %-20s%-10.7f\n" string(key) energies[key]
end
@printf "\n    %-20s%-15.12f\n\n" "total" sum(values(energies))

# Plot band structure
# plot_bands(ham, n_bands_plot, kline_density, atoms, scfres.εF).show()
