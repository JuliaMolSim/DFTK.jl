# # Phonon computations
#
# This is a quick sketch how to run a simple phonon calculation using DFTK.

# !!! warning "Preliminary implementation"
#     Practical phonon computations have only seen rudimentary testing as of now.
#     As of now we do not yet recommend relying on this feature for production
#     calculations. Some of the limitations are:
#     - symmetries must be disabled (pass `symmetries=false` to the model)
#     - only LDA functionals are supported
#     - non-linear core corrections from the pseudopotentials are not supported
#     - MPI parallelization over k-points is not supported (due to ``k`` and ``k+q`` interactions)
#     We appreciate any issues, bug reports or PRs.
#
# First we run an SCF calculation.

using AtomsBuilder
using DFTK
using Printf
using PseudoPotentialData

pseudopotentials = PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth")
## Make sure to disable symmetries:
model  = model_DFT(bulk(:Si); pseudopotentials, functionals=LDA(), symmetries=false)
basis  = PlaneWaveBasis(model; Ecut=10, kgrid=[4, 4, 4])
scfres = self_consistent_field(basis, tol=1e-8);
nothing  # hide

# Next we compute the phonon modes at the q-point `[1/4, 1/4, 1/4]`.

phret_q0 = @time DFTK.phonon_modes(scfres; q=[0.25, 0.25, 0.25]);
nothing  # hide

# These are the final phonon frequencies:
for (i, ω) in enumerate(phret_q0.frequencies)
    @printf("Mode %2d: %8.3f cm-1\n", i, ω .* DFTK.hartree_to_cm⁻¹)
end
