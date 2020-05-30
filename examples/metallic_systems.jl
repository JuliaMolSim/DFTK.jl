# # Temperature and metallic systems
#
# In this example we consider the modeling of a magnesium lattice
# as a simple example for a metallic system.
# For our treatment we will use the PBE exchange-correlation functional.
# First we import required packages and setup the lattice.

using DFTK
using Plots

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

kspacing = 0.5      # Minimal spacing of k-points
Ecut = 5            # kinetic energy cutoff in Hartree
temperature = 0.01  # Smearing temperature in Hartree

kgrid = kgrid_size_from_minimal_spacing(lattice, kspacing)
kgrid_size_from_minimal_spacing(lattice, kspacing)
model = model_DFT(lattice, atoms, [:gga_x_pbe, :gga_c_pbe];
                  temperature=temperature,
                  smearing=DFTK.Smearing.FermiDirac())
basis = PlaneWaveBasis(model, Ecut, kgrid=kgrid);

# Finally we run the SCF. Even though two magnesium atoms in
# our pseudopotential model only result in four valence electrons being explicitly
# treated, we still solve for eight bands in order to capture
# the partial occupations beyond the Fermi level due to
# the employed smearing scheme.
scfres = self_consistent_field(basis, n_bands=8);
#-
scfres.energies

# The fact that magnesium is a metal is confirmed
# by plotting the density of states around the Fermi level.

εs = range(minimum(minimum(scfres.eigenvalues)) - .5,
           maximum(maximum(scfres.eigenvalues)) + .5, length=1000)
Ds = DOS.(εs, Ref(basis), Ref(scfres.eigenvalues))
q = plot(εs, Ds, label="DOS")
vline!(q, [scfres.εF], label="εF")
