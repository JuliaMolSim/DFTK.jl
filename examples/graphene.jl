using DFTK
using Printf

#
# Example of a medium-scale graphene calculation. Only suitable for running
# on a cluster or machine with large memory.
## tags: long
#

kgrid = [12, 12, 4]
Tsmear = 0.0009500431544769484
Ecut = 35

lattice = [4.659533614391621 -2.3297668071958104 0.0;
           0.0 4.035274479829987 0.0;
           0.0 0.0 15.117809010356462]
C = Species(6, load_psp("hgh/pbe/c-q4"))
composition = [C => [[0.0, 0.0, 0.0], [0.33333333333, 0.66666666667, 0.0]]]

model = model_dft(lattice, [:gga_x_pbe, :gga_c_pbe], composition...;
                  temperature=Tsmear, smearing=DFTK.Smearing.Gaussian())
kcoords, ksymops = bzmesh_ir_wedge(kgrid, lattice, composition...)
basis = PlaneWaveBasis(model, Ecut, kcoords, ksymops)

# Run SCF
n_bands = 6
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
