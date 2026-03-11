# # Phonon computations
#
# This is a quick sketch how to run a simple phonon calculation using DFTK.

# !!! warning "Preliminary implementation"
#     Practical phonon computations have only seen rudimentary testing as of now.
#     As of now we do not yet recommend relying on this feature for production
#     calculations. We appreciate any issues, bug reports or PRs.
#
# First we run an SCF calculation.

using AtomsBuilder
using DFTK
using PseudoPotentialData

pseudopotentials = PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth")
model  = model_DFT(bulk(:Si); pseudopotentials, functionals=LDA())
basis  = PlaneWaveBasis(model; Ecut=10, kgrid=[4, 4, 4])
scfres = self_consistent_field(basis, tol=1e-8);
nothing  # hide

# Next we compute the phonon modes at the q-point `[1/4, 1/4, 1/4]`.

scfres = DFTK.unfold_bz(scfres)
phret_q0 = @time DFTK.phonon_modes(scfres; q=[0.25, 0.25, 0.25]);
nothing  # hide

# These are the final phonon frequencies:

phret_q0.frequencies
