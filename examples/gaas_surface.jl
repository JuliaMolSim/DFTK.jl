# # Modelling a gallium arsenide surface
#
# This example shows how to use the atomistic simulation environment or ASE for short,
# to set up and run a particular calculation of a gallium arsenide surface.
# ASE is a Python package to simplify the process of setting up,
# running and analysing results from atomistic simulations across different simulation codes.
# For more details on the integration DFTK provides with ASE,
# see [Atomistic simulation environment](@ref).
#
# In this example we will consider modelling the (1, 1, 0) GaAs surface separated by vacuum.

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
ase.io.write("surface.png", surface * pytuple((3, 3, 1)), rotation="-90x, 30y, -75z")

#md # ```@raw html
#md # <img src="../surface.png" width=500 height=500 />
#md # ```
#nb # <img src="https://docs.dftk.org/stable/surface.png" width=500 height=500 />

# Use the `pyconvert` function from `PythonCall` to convert the ASE atoms
# to an AtomsBase-compatible system.
# This can then be used in the same way as other `AtomsBase` systems
# (see [AtomsBase integration](@ref) for details) to construct a DFTK model:

using DFTK

pseudopotentials = Dict(:Ga => "hgh/pbe/ga-q3.hgh",
                        :As => "hgh/pbe/as-q5.hgh")
model = model_DFT(pyconvert(AbstractSystem, surface);
                  functionals=PBE(),
                  temperature=1e-3,
                  smearing=DFTK.Smearing.Gaussian(),
                  pseudopotentials)

# In the above we use the `pseudopotential` keyword argument to
# assign the respective pseudopotentials to the imported `model.atoms`.
# Try lowering the SCF convergence tolerance (`tol`)
# or try `mixing=KerkerMixing()` to see the full challenge of this system.

basis  = PlaneWaveBasis(model; Ecut, kgrid)
scfres = self_consistent_field(basis; tol=1e-6, mixing=LdosMixing());
#-
scfres.energies
