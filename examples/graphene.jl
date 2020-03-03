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
C = ElementPsp(:C, load_psp("hgh/pbe/c-q4"))
atoms = [C => [[0.0, 0.0, 0.0], [0.33333333333, 0.66666666667, 0.0]]]

model = model_dft(lattice, [:gga_x_pbe, :gga_c_pbe], atoms;
                  temperature=Tsmear, smearing=DFTK.Smearing.Gaussian())
basis = PlaneWaveBasis(model, Ecut, kgrid=kgrid)

# Run SCF
n_bands = 6
ham = Hamiltonian(basis, guess_density(basis))
scfres = self_consistent_field(ham, n_bands)

print_energies(scfres.energies)
