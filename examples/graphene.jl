using DFTK

#
# Example of a medium-scale graphene calculation. Only suitable for running
# on a cluster or machine with large memory.
## tags: long
#

kgrid = [12, 12, 4]
Tsmear = 0.0009500431544769484
Ecut = 15

lattice = [4.659533614391621 -2.3297668071958104 0.0;
           0.0 4.035274479829987 0.0;
           0.0 0.0 15.117809010356462]
C = ElementPsp(:C, load_psp("hgh/pbe/c-q4"))
atoms = [C => [[0.0, 0.0, 0.0], [0.33333333333, 0.66666666667, 0.0]]]

model = model_DFT(lattice, atoms, [:gga_x_pbe, :gga_c_pbe];
                  temperature=Tsmear, smearing=Smearing.Gaussian())
basis = PlaneWaveBasis(model, Ecut, kgrid=kgrid)

# Run SCF
n_bands = 6
scfres = self_consistent_field(basis; n_bands=n_bands)

# Print obtained energies
println()
display(scfres.energies)
