using DFTK
using Printf
using PyCall
import PyPlot

# Calculation parameters
kgrid = [4, 4, 4]        # k-Point grid
Ecut = 15                # kinetic energy cutoff in Hartree
supercell = [1, 1, 1]    # Lattice supercell
n_bands = 8              # Number of bands for SCF and plotting
Tsmear = 0.01            # Smearing temperature in Hartree
kline_density = 20       # Density of k-Points for bandstructure

# Setup magnesium lattice (constants in Bohr)
a = 3.0179389193174084
b = 5.227223542397263
c = 9.773621942589742
lattice = [[-a -a  0]; [-b  b  0]; [0   0 -c]]
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

# Plot band structure
plot_bands(ham, n_bands, kline_density, composition, scfres.εF).show()

# Plot DOS
εs = range(minimum(minimum(scfres.orben)) - 1, maximum(maximum(scfres.orben)) + 1, length=1000)
Ds = DOS.(εs, Ref(basis), Ref(scfres.orben), T=Tsmear*4, smearing=DFTK.smearing_methfessel_paxton_1)
PyPlot.plot(εs, Ds)
PyPlot.axvline(scfres.εF)
