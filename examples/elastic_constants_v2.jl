# # Elastic constants

# We compute *clamped-ion* elastic constants of a crystal using
# the algorithmic differentiation density-functional perturbation theory (AD-DFPT) approach
# as introduced in [^SPH25].
#
# [^SPH25]:
#     Schmitz, N. F., Ploumhans, B., & Herbst, M. F. (2026)
#     *Algorithmic differentiation for plane-wave DFT: materials design, error control and learning model parameters.*
#     [npj Computational Materials **12**, 6 (2026)](https://doi.org/10.1038/s41524-025-01880-3).
#     ([Supplementary material and computational scripts](https://github.com/niklasschmitz/ad-dfpt)).
#
# We consider a crystal in its equilibrium configuration, where all atomic forces
# and stresses vanish.  Homogeneous strains $η$ are then applied
# relative to this relaxed structure.
# The elastic constants are derived from the stress-strain relationship.
# In [Voigt notation](https://en.wikipedia.org/wiki/Voigt_notation),
# the stress $\sigma$ and strain $\eta$ tensors are represented as 6-component vectors.
# The elastic constants $C$ are then given by
# the Jacobian of the stress with respect to strain, forming a $6 \times 6$ matrix
# ```math
#   C = \frac{\partial \sigma}{\partial \eta}.
# ```
#
# This example computes the *clamped-ion* elastic tensor, keeping internal
# atomic positions fixed under strain.  The *relaxed-ion* tensor includes
# additional corrections from internal relaxations, which can be obtained
# from first-order atomic displacements in DFPT (see [^Wu2005]).
#
# [^Wu2005]:
#     Wu, X., Vanderbilt, D., & Hamann, D. R. (2005).
#     *Systematic treatment of displacements, strains, and electric fields in density-functional perturbation theory.*
#     [Physical Review B, 72(3), 035105](https://doi.org/10.1103/PhysRevB.72.035105).


using DFTK
using PseudoPotentialData
using LinearAlgebra
using ForwardDiff
using DifferentiationInterface
using AtomsBuilder
using Unitful
using UnitfulAtomic

# ## Computing PBE elastic constants
#
# We start with the PBE [^PBE1996] functional.
#
# [^PBE1996]:
#     Perdew, J. P., Burke, K., & Ernzerhof, M. (1996).
#     *Generalized Gradient Approximation Made Simple.*
#     [Physical Review Letters, 77(18), 3865-3868](https://doi.org/10.1103/PhysRevLett.77.3865).

pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
a0_pbe = 10.33u"bohr"  # Equilibrium lattice constant of silicon with PBE
model0 = model_DFT(bulk(:Si; a=a0_pbe); pseudopotentials, functionals=PBE())

Ecut = recommended_cutoff(model0).Ecut
kgrid = [4, 4, 4]
tol = 1e-6

# The `elastic_tensor` postprocessing function automatically detects crystal
# symmetry and chooses the appropriate strain patterns to extract all
# independent elastic constants:

basis = PlaneWaveBasis(model0; Ecut, kgrid)
scfres = self_consistent_field(basis; tol)
(; C, dfpt_res) = DFTK.elastic_tensor_v2(scfres)

println("C11: ", uconvert(u"GPa", C[1, 1] * u"hartree" / u"bohr"^3))
println("C12: ", uconvert(u"GPa", C[1, 2] * u"hartree" / u"bohr"^3))
println("C44: ", uconvert(u"GPa", C[4, 4] * u"hartree" / u"bohr"^3))
