using DFTK
using PyCall

#
# Example using ASE to build a (1, 1, 0) GaAs surface separated by vacuum.
# See https://wiki.fysik.dtu.dk/ase/install.html for details
## tags: long
#

miller = (1, 1, 0)              # Surface Miller indices
n_GaAs = 20                     # Number of GaAs layers
n_vacuum = 20                   # Number of vacuum layers
Ecut = 15                       # Hartree
kgrid = (4, 4, 1)               # Monkhorst-Pack mesh
# kgrid = (13, 13, 1)           # Better Monkhorst-Pack mesh

# Use ASE to build the structure
ase_build = pyimport("ase.build")
gaas = ase_build.bulk("GaAs", "zincblende", a=2.82685)
surface = ase_build.surface(gaas, miller, n_GaAs, 0, periodic=true)

# Get the amount of vacuum in Ǎngström we need to add
d_vacuum = maximum(maximum, surface.cell) / n_GaAs * n_vacuum
surface = ase_build.surface(gaas, miller, n_GaAs, d_vacuum, periodic=true)


# Convert to DFTK datastructures and run calculation
atoms = load_atoms(surface)
atoms = [ElementPsp(el.symbol, psp=load_psp(el.symbol, functional="pbe")) => position
         for (el, position) in atoms]

lattice = load_lattice(surface)
model = model_DFT(lattice, atoms, [:gga_x_pbe, :gga_c_pbe])
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)

scfres = self_consistent_field(basis, tol=1e-8,
                               n_bands=ceil(Int, model.n_electrons / 2),
                               mixing=KerkerMixing())

# Print obtained energies
println()
display(scfres.energies)
