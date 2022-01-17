# # Temperature and metallic systems
#
# In this example we consider the modeling of a magnesium lattice
# as a simple example for a metallic system.
# For our treatment we will use the PBE exchange-correlation functional.
# First we import required packages and setup the lattice.
# Again notice that DFTK uses the convention that lattice vectors are
# specified column by column.

using DFTK
using Plots
using Unitful
using UnitfulAtomic

a = 3.01794  # bohr
b = 5.22722  # bohr
c = 9.77362  # bohr
lattice = [[-a -a  0]; [-b  b  0]; [0   0 -c]]
Mg = ElementPsp(:Mg, psp=load_psp("hgh/pbe/Mg-q2"))
atoms = [Mg => [[2/3, 1/3, 1/4], [1/3, 2/3, 3/4]]];

# Next we build the PBE model and discretize it.
# Since magnesium is a metal we apply a small smearing
# temperature to ease convergence using the Fermi-Dirac
# smearing scheme. Note that both the `Ecut` is too small
# as well as the minimal ``k``-point spacing
# `kspacing` far too large to give a converged result.
# These have been selected to obtain a fast execution time.
# By default `PlaneWaveBasis` chooses a `kspacing`
# of `2π * 0.022` inverse Bohrs, which is much more reasonable.

kspacing = 0.945 / u"angstrom"        # Minimal spacing of k-points, 
##                                      in units of wavevectors (inverse Bohrs)
Ecut = 5                              # Kinetic energy cutoff in Hartree
temperature = 0.01                    # Smearing temperature in Hartree
smearing = DFTK.Smearing.FermiDirac() # Smearing method 
##                                      also supported: Gaussian,
##                                      MarzariVanderbilt,
##                                      and MethfesselPaxton(order)

model = model_DFT(lattice, atoms, [:gga_x_pbe, :gga_c_pbe];
                  temperature=temperature,
                  smearing=smearing)
kgrid = kgrid_from_minimal_spacing(lattice, kspacing)
basis = PlaneWaveBasis(model; Ecut, kgrid);

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
plot_dos(scfres)
