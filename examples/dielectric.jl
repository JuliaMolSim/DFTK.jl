# # Dielectric tensor and polarizability
#
# This example computes the electronic (clamped-ion) dielectric tensor ``ε∞`` of silicon,
# i.e. the response of the insulator to a homogeneous electric field, using
# density-functional perturbation theory (the "d/dk" trick of the modern theory of
# polarization).

# !!! warning "Preliminary implementation — insulators only"
#     This feature has only seen rudimentary testing as of now, and we do not yet recommend
#     relying on it for production calculations. Current limitations:
#     - insulators only (the metallic intraband/Drude term is not included; the routine
#       errors on fractional occupations)
#     - symmetries must be disabled (pass `symmetries=false` to the model)
#     - spin-unpolarized (`:none`) only
#     We appreciate any issues, bug reports or PRs.
#
# First we run an SCF calculation. Note that a dielectric response is a Brillouin-zone
# integral that converges slowly, so a much denser `kgrid` than for a total energy is needed
# for a well-converged number.

using AtomsBuilder
using DFTK
using PseudoPotentialData

pseudopotentials = PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth")
## Make sure to disable symmetries:
model  = model_DFT(bulk(:Si); pseudopotentials, functionals=LDA(), symmetries=false)
basis  = PlaneWaveBasis(model; Ecut=12, kgrid=[4, 4, 4])
scfres = self_consistent_field(basis; tol=1e-8);
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
