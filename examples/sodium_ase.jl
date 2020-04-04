using DFTK
using PyCall

# This example requires ASE to be installed in your python environment.
# See https://wiki.fysik.dtu.dk/ase/install.html for details
ase_build = pyimport("ase.build")

# Calculation parameters
Ecut = 15                            # kinetic energy cutoff in Hartree
n_bands = 14                         # number of bands to plot in the bandstructure
kspacing = 2π * 0.04 / DFTK.units.Ǎ  # Minimal spacing between k-points
Tsmear = 0.01                        # Smearing temperature in Hartree

# Use ASE to make a cubic sodium lattice
ase_na = ase_build.bulk("Na", cubic=true)
lattice = load_lattice(ase_na)
atoms = load_atoms(ase_na)

# The result of `load_atoms` does not contain pseudopotential information
# if ase.Atoms are used as the argument. The next line attaches PBE pseudopotentials
atoms = [ElementPsp(el.symbol, psp=load_psp(el.symbol, functional="pbe")) => position
         for (el, position) in atoms]

model = model_DFT(lattice, atoms, [:gga_x_pbe, :gga_c_pbe],
                  temperature=Tsmear, smearing=DFTK.Smearing.FermiDirac())
basis = PlaneWaveBasis(model, Ecut;
                       kgrid=kgrid_size_from_minimal_spacing(lattice, kspacing))

scfres = self_consistent_field(basis, n_bands=n_bands, tol=1e-4)

# Print obtained energies
println()
display(scfres.energies)
