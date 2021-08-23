#
# Example of a medium-scale graphene calculation. Only suitable for running
# on a cluster or machine with large memory.
#src tags: long
#

using DFTK
setup_threading()

kgrid = [12, 12, 4]
Ecut = 15

lattice = [4.66 -2.33 0.00;
           0.00  4.04 0.00
           0.00  0.00 15.12]
C = ElementPsp(:C, psp=load_psp("hgh/pbe/c-q4"))
atoms = [C => [[0.0, 0.0, 0.0], [1//3, 2//3, 0.0]]]

model = model_PBE(lattice, atoms; temperature=1e-3, smearing=Smearing.Gaussian())
basis = PlaneWaveBasis(model; Ecut, kgrid)

# Run SCF
scfres = self_consistent_field(basis)

println("Carbon forces:")
println(compute_forces(scfres)[1])

# Print obtained energies
println()
println(scfres.energies)
