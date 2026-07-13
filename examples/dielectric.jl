# # Dielectric tensor and polarizability
#
# This example computes the electronic (clamped-ion) dielectric tensor ``ε∞`` of silicon,
# i.e. the response of the insulator to a homogeneous electric field, using
# density-functional perturbation theory (the "d/dk" trick of the modern theory of
# polarization).

# First we run an SCF calculation. Note that a much denser `kgrid`
# than for a total energy is needed for a well-converged number. We keep the crystal
# symmetries switched on here so that the SCF only samples the irreducible wedge of the
# Brillouin zone, which is considerably cheaper.

using AtomsBuilder
using DFTK
using PseudoPotentialData

pseudopotentials = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
model  = model_DFT(bulk(:Si); pseudopotentials, functionals=LDA())
basis  = PlaneWaveBasis(model; Ecut=12, kgrid=[4, 4, 4])
scfres = self_consistent_field(basis; tol=1e-8);
nothing  # hide

# The electric-field response breaks the k-point symmetries (a symmetry-reduced basis
# would over-symmetrize the field-induced density change to zero), so before computing it
# we unfold the result to the full Brillouin zone:

scfres = DFTK.unfold_bz(scfres);
nothing  # hide

# Then we compute the dielectric tensor and polarizability by linear response:

diel = compute_dielectric(scfres);
nothing  # hide

# The dielectric tensor of cubic silicon is isotropic, `ε∞ ≈ ε·I`:

diel.ε∞

# The value here is not converged (the grid above is far too coarse); denser `kgrid`s bring it
# down towards the experimental `ε∞ ≈ 11.7` (LDA slightly overestimates, giving `≈ 13`).
#
# The static per-cell polarizability `Ω·χ` (`P = χ·E`) is returned as well:

diel.polarizability
