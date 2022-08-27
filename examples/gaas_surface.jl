# # Modelling a gallium arsenide surface
#
# This example shows how to use the
# [atomistic simulation environment](https://wiki.fysik.dtu.dk/ase/index.html),
# or ASE for short,
# to set up a particular gallium arsenide surface
# and run the resulting calculation in DFTK.
# The particular example we consider the (1, 1, 0) GaAs surface separated by vacuum.
#

# Parameters of the calculation. Since this surface is far from easy to converge,
# we made the problem simpler by choosing a smaller `Ecut` and smaller values
# for `n_GaAs` and `n_vacuum`.
# More interesting settings are `Ecut = 15` and `n_GaAs = n_vacuum = 20`.
#
miller = (1, 1, 0)   # Surface Miller indices
n_GaAs = 2           # Number of GaAs layers
n_vacuum = 4         # Number of vacuum layers
Ecut = 5             # Hartree
kgrid = (4, 4, 1);   # Monkhorst-Pack mesh

# Use ASE to build the structure:
using PyCall

ase_build = pyimport("ase.build")
a = 5.6537  # GaAs lattice parameter in Ångström (because ASE uses Å as length unit)
gaas = ase_build.bulk("GaAs", "zincblende", a=a)
surface = ase_build.surface(gaas, miller, n_GaAs, 0, periodic=true);

# Get the amount of vacuum in Ångström we need to add
d_vacuum = maximum(maximum, surface.cell) / n_GaAs * n_vacuum
surface = ase_build.surface(gaas, miller, n_GaAs, d_vacuum, periodic=true);

# Write an image of the surface and embed it as a nice illustration:
pyimport("ase.io").write("surface.png", surface * (3, 3, 1),
                         rotation="-90x, 30y, -75z")

#md # ```@raw html
#md # <img src="../surface.png" width=500 height=500 />
#md # ```
#nb # <img src="https://docs.dftk.org/stable/surface.png" width=500 height=500 />

# Use the `load_atoms`, `load_positions` and `load_lattice` functions
# to convert to DFTK datastructures.
# These two functions not only support importing ASE atoms into DFTK,
# but a few more third-party datastructures as well.
# Typically the imported `atoms` use a bare Coulomb potential,
# such that appropriate pseudopotentials need to be attached in a post-step:

using DFTK

positions = load_positions(surface)
lattice   = load_lattice(surface)
atoms = map(load_atoms(surface)) do el
    if el.symbol == :Ga
        ElementPsp(:Ga, psp=load_psp("hgh/pbe/ga-q3.hgh"))
    elseif el.symbol == :As
        ElementPsp(:As, psp=load_psp("hgh/pbe/as-q5.hgh"))
    else
        error("Unsupported element: $el")
    end
end;

# We model this surface with (quite large a) temperature of 0.01 Hartree
# to ease convergence. Try lowering the SCF convergence tolerance (`tol`)
# or the `temperature` or try `mixing=KerkerMixing()`
# to see the full challenge of this system.
model = model_DFT(lattice, atoms, positions, [:gga_x_pbe, :gga_c_pbe],
                  temperature=0.001, smearing=DFTK.Smearing.Gaussian())
basis = PlaneWaveBasis(model; Ecut, kgrid)

scfres = self_consistent_field(basis, tol=1e-4, mixing=LdosMixing());
#-
scfres.energies
