# # [Temperature and metallic systems](@id metallic-systems)
#
# In this example we consider the modeling of a magnesium lattice
# as a simple example for a metallic system.
# For our treatment we will use the PBE exchange-correlation functional.
# First we import required packages and setup the lattice.
# Again notice that DFTK uses the convention that lattice vectors are
# specified column by column.

using DFTK
using Plots
using PseudoPotentialData
using Unitful
using UnitfulAtomic

a = 3.01794  # Bohr
b = 5.22722  # Bohr
c = 9.77362  # Bohr
lattice = [[-a -a  0]; [-b  b  0]; [0   0 -c]]

pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
Mg = ElementPsp(:Mg, pseudopotentials)
atoms     = [Mg, Mg]
positions = [[2/3, 1/3, 1/4], [1/3, 2/3, 3/4]];

# Next we build the PBE model and discretize it.
# Since magnesium is a metal we apply a small smearing
# temperature to ease convergence using the Fermi-Dirac
# smearing scheme.

kgrid = KgridSpacing(0.9 / u"Å")    # Minimal spacing of k-points,
##                                      in units of wavevectors (inverse Bohrs)
temperature = 0.01                    # Smearing temperature in Hartree
Ecut = 10                             # Kinetic energy cutoff in Hartree
smearing = DFTK.Smearing.FermiDirac() # Smearing method
##                                      also supported: Gaussian,
##                                      MarzariVanderbilt,
##                                      and MethfesselPaxton(order)

model = model_DFT(lattice, atoms, positions;
                  functionals=[:gga_x_pbe, :gga_c_pbe], temperature, smearing)
basis = PlaneWaveBasis(model; Ecut, kgrid);

# Note, that in this example both the `Ecut` as well as the minimal ``k``-point spacing
# `0.9 / u"Å"` far too large to give a converged result. In the online documentation we
# have used these small values to obtain a fast execution time.

# !!! note "Ecut and kgrid are optional"
#     Both the `Ecut` and the `kgrid` keyword argument in `PlaneWaveBasis`
#     are optional. If the user does not specify these values, DFTK will
#     try to determine reasonable default values:
#     - Kgrid default:`kgrid=KgridSpacing(2π * 0.022 / u"bohr")`, which
#       usually gives reasonable results for a first calculation.
#     - Ecut default: DFTK will consult the [PseudoPotentialData.jl](https://github.com/JuliaMolSim/PseudoPotentialData.jl)
#       library for a recommended kinetic energy cutoff and use the maximal
#       value over all atoms of the calculation.   See the [Pseudopotentials](@ref)
#       chapter for more details on using pseudopotentials with DFTK.
#       For cases where no recommended values can be determined,
#       DFTK will throw an error and expects the user to manually provide
#       a value for `Ecut`.
#     Therefore we could also construct a more reasonable basis as follows:

basis_default = PlaneWaveBasis(model)

# As can be seen the default discretisation selects the much finer discretisation
# parameters `Ecut=42` and `kgrid=[9, 9, 5]`.

# Finally we run the SCF. Two magnesium atoms in
# our pseudopotential model result in four valence electrons being explicitly
# treated. Nevertheless this SCF will solve for eight bands by default
# in order to capture partial occupations beyond the Fermi level due to
# the employed smearing scheme. In this example we use a damping of `0.8`.
# The default `LdosMixing` should be suitable to converge metallic systems
# like the one we model here. For the sake of demonstration we still switch to
# Kerker mixing here.
scfres = self_consistent_field(basis, damping=0.8, mixing=KerkerMixing());
#-
scfres.occupation[1]
#-
scfres.energies

# The fact that magnesium is a metal is confirmed
# by plotting the density of states around the Fermi level.
# To get better plots, we decrease the k spacing a bit for this step,
# i.e. we use a finer k-point mesh with more points.
bands = compute_bands(scfres, KgridSpacing(0.7 / u"Å"))
plot_dos(bands)
