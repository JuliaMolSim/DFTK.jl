# # Modelling a gallium arsenide surface
#
# This example shows how to use the
# [atomistic simulation environment](https://wiki.fysik.dtu.dk/ase/index.html),
# or ASE for short,
# to set up and run a particular calculation of a gallium arsenide surface.
# ASE is a Python package which intents to simplify the process of setting up,
# running and analysing results from atomistic simulations across different simulation codes.
# By means of [ASEconvert](https://github.com/mfherbst/ASEconvert.jl) it is seamlessly
# integrated with the AtomsBase ecosystem and thus available to DFTK via our own
# [AtomsBase integration](@ref).
#
# In this example we will consider modelling the (1, 1, 0) GaAs surface separated by vacuum
# using density-functional theory.

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
using ASEconvert

a = 5.6537  # GaAs lattice parameter in Ångström (because ASE uses Å as length unit)
gaas = ase.build.bulk("GaAs", "zincblende"; a)
surface = ase.build.surface(gaas, miller, n_GaAs, 0, periodic=true);

# Get the amount of vacuum in Ångström we need to add
d_vacuum = maximum(maximum, surface.cell) / n_GaAs * n_vacuum
surface = ase.build.surface(gaas, miller, n_GaAs, d_vacuum, periodic=true);

# Write an image of the surface and embed it as a nice illustration:
ase.io.write("surface.png", surface * (3, 3, 1), rotation="-90x, 30y, -75z")

#md # ```@raw html
#md # <img src="../surface.png" width=500 height=500 />
#md # ```
#nb # <img src="https://docs.dftk.org/stable/surface.png" width=500 height=500 />

# Use the `pyconvert` function from `PythonCall` to convert to an AtomsBase-compatible system.
# These two functions not only support importing ASE atoms into DFTK,
# but a few more third-party data structures as well.
# Typically the imported `atoms` use a bare Coulomb potential,
# such that appropriate pseudopotentials need to be attached in a post-step:

using DFTK
system = attach_psp(pyconvert(AbstractSystem, surface);
                    Ga=load_psp("hgh/pbe/ga-q3.hgh"),
                    As=load_psp("hgh/pbe/as-q5.hgh"))

# We model this surface with (quite large a) temperature of 0.01 Hartree
# to ease convergence. Try lowering the SCF convergence tolerance (`tol`)
# or the `temperature` or try `mixing=KerkerMixing()`
# to see the full challenge of this system.
model = model_DFT(system, [:gga_x_pbe, :gga_c_pbe],
                  temperature=0.001, smearing=DFTK.Smearing.Gaussian())
basis = PlaneWaveBasis(model; Ecut, kgrid)

scfres = self_consistent_field(basis, tol=1e-4, mixing=LdosMixing());
#-
scfres.energies
