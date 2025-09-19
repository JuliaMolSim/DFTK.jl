# # Collinear spin and magnetic systems
#
# In this example we consider iron in the BCC phase.
# To show that this material is ferromagnetic we will model it once
# allowing collinear spin polarization and once without
# and compare the resulting SCF energies. In particular
# the ground state can only be found if collinear spins are allowed.
#
# The `bulk(:Fe)` function from `AtomsBuilder` returns a BCC iron setup
# with a single iron atom inside the unit cell.

using AtomsBuilder
using PseudoPotentialData
using DFTK

bulk(:Fe)

# First we consider a setup without spin polarization.
# To get the ground-state energy of this system we use an LDA model
# and rather moderate discretisation parameters.

Ecut  = 15         # kinetic energy cutoff in Hartree
kgrid = [3, 3, 3]  # k-point grid (Regular Monkhorst-Pack grid)
pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")

model_nospin  = model_DFT(bulk(:Fe); pseudopotentials, functionals=LDA(), temperature=0.01)
basis_nospin  = PlaneWaveBasis(model_nospin; kgrid, Ecut)
scfres_nospin = self_consistent_field(basis_nospin; tol=1e-4, mixing=KerkerDosMixing());
#-
scfres_nospin.energies

# Since we did not specify any initial magnetic moment on the iron atom,
# DFTK will automatically assume that a calculation with only spin-paired
# electrons should be performed. As a result the obtained ground state
# features no spin-polarization.

# Now we repeat the calculation, but give the iron atom an initial magnetic moment.
# For specifying the magnetic moment pass the desired excess of spin-up over spin-down
# electrons at each centre to the `Model` and the guess density functions.
# In this case we seek the state with as many spin-parallel
# ``d``-electrons as possible. In our pseudopotential model the 8 valence
# electrons are 1 pair of ``s``-electrons, 1 pair of ``d``-electrons
# and 4 unpaired ``d``-electrons giving a desired magnetic moment of `4` at the iron centre.
# The structure (i.e. pair mapping and order) of the `magnetic_moments` array needs to agree
# with the `atoms` array and `0` magnetic moments need to be specified as well.

magnetic_moments = [4];

# !!! tip "Units of the magnetisation and magnetic moments in DFTK"
#     Unlike all other quantities magnetisation and magnetic moments in DFTK
#     are given in units of the Bohr magneton ``μ_B``, which in atomic units has the
#     value ``\frac{1}{2}``. Since ``μ_B`` is (roughly) the magnetic moment of
#     a single electron the advantage is that one can directly think of these
#     quantities as the excess of spin-up electrons or spin-up electron density.
#
# We repeat the calculation using the same model as before. DFTK now detects
# the non-zero moment and switches to a collinear calculation.

model = model_DFT(bulk(:Fe); pseudopotentials, functionals=LDA(),
                  temperature=0.01, magnetic_moments)
basis = PlaneWaveBasis(model; Ecut, kgrid)
ρ0 = guess_density(basis, magnetic_moments)
scfres = self_consistent_field(basis, tol=1e-6; ρ=ρ0, mixing=KerkerDosMixing());
#-
scfres.energies

# !!! note "Model and magnetic moments"
#     DFTK does not store the `magnetic_moments` inside the `Model`, but only uses them
#     to determine the lattice symmetries. This step was taken to keep `Model`
#     (which contains the physical model) independent of the details of the numerical details
#     such as the initial guess for the spin density.
#
# In direct comparison we notice the first, spin-paired calculation to be
# a little higher in energy
println("No magnetization: ", scfres_nospin.energies.total)
println("Magnetic case:    ", scfres.energies.total)
println("Difference:       ", scfres.energies.total - scfres_nospin.energies.total);
# Notice that with the small cutoffs we use to generate the online
# documentation the calculation is far from converged.
# With more realistic parameters a larger energy difference of about
# 0.1 Hartree is obtained.

# The spin polarization in the magnetic case is visible if we
# consider the occupation of the spin-up and spin-down Kohn-Sham orbitals.
# Especially for the ``d``-orbitals these differ rather drastically.
# For example for the first ``k``-point:

iup   = 1
idown = iup + length(scfres.basis.kpoints) ÷ 2
@show scfres.occupation[iup][1:7]
@show scfres.occupation[idown][1:7];

# Similarly the eigenvalues differ
@show scfres.eigenvalues[iup][1:7]
@show scfres.eigenvalues[idown][1:7];

# !!! note "``k``-points in collinear calculations"
#     For collinear calculations the `kpoints` field of the `PlaneWaveBasis` object contains
#     each ``k``-point coordinate twice, once associated with spin-up and once with down-down.
#     The list first contains all spin-up ``k``-points and then all spin-down ``k``-points,
#     such that `iup` and `idown` index the same ``k``-point, but differing spins.

# We can observe the spin-polarization by looking at the density of states (DOS)
# around the Fermi level, where the spin-up and spin-down DOS differ.
using Plots
bands_666 = compute_bands(scfres, MonkhorstPack(6, 6, 6))  # Increase kgrid to get nicer DOS.
plot_dos(bands_666)
# Note that if same k-grid as SCF should be employed, a simple `plot_dos(scfres)`
# is sufficient.
# We can clearly see that the origin of this spin-polarization traces back to the 
# 3D orbital contribution if we look at the corresponding projected density of states (PDOS).
plot_pdos(bands_666; iatom=1, label="3D")

# Similarly the band structure shows clear differences between both spin components.
using Unitful
using UnitfulAtomic
bands_kpath = compute_bands(scfres; kline_density=6)
plot_bandstructure(bands_kpath)
