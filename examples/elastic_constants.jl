# # Elastic constants

# We compute *clamped-ion* elastic constants of a crystal using the algorithmic differentiation 
# density-functional perturbation theory (AD-DFPT) approach as introduced in [^SPH25].
# 
# [^SPH25]: 
#     Schmitz, N. F., Ploumhans, B., & Herbst, M. F. (2025)
#     *Algorithmic differentiation for plane-wave DFT: materials design, error control and learning model parameters.*
#     [arXiv:2509.07785](https://arxiv.org/abs/2509.07785)

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
# The sparsity pattern of the matrix $C$ follows from crystal symmetry
# and is tabulated in standard references (eg. Table 9 in [^Nye1985]).
# This sparsity can be used a priori to reduce the number of strain patterns
# that need to be probed to extract all independent components of $C$.
# For example, cubic crystals have only three independent elastic constants $C_{11}$, $C_{12}$ and $C_{44}$,
# with the pattern
# ```math
# C = \begin{pmatrix}
#   C_{11} & C_{12} & C_{12} & 0      & 0      & 0 \\
#   C_{12} & C_{11} & C_{12} & 0      & 0      & 0 \\
#   C_{12} & C_{12} & C_{11} & 0      & 0      & 0 \\
#   0      & 0      & 0      & C_{44} & 0      & 0 \\
#   0      & 0      & 0      & 0      & C_{44} & 0 \\
#   0      & 0      & 0      & 0      & 0      & C_{44} \\
# \end{pmatrix}.
# ```
# Thus we can just choose a suitable strain pattern $\dot{\eta} = (1, 0, 0, 1, 0, 0)^\top$,
# such that $C\dot{\eta} = (C_{11}, C_{12}, C_{12}, C_{44}, 0, 0)^\top$. That is,
# for cubic crystals like diamond silicon we obtain all independent elastic
# constants from a single Jacobian-vector product on the stress-strain function.
#
# [^Nye1985]:
#      Nye, J. F. (1985).
#      *Physical Properties of Crystals*. Oxford University Press.
#      Comment: Since the elastic tensor transforms equivariantly under rotations,
#      its numerical components depend on the chosen Cartesian coordinate frame.
#      These tabulated patterns assume a standardized orientation of the structure
#      with respect to conventional crystallographic axes.
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


pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
a0_pbe = 10.33u"bohr"  # Equilibrium lattice constant of silicon with PBE
model0 = model_DFT(bulk(:Si; a=a0_pbe); pseudopotentials, functionals=PBE())

Ecut = recommended_cutoff(model0).Ecut
kgrid = [4, 4, 4]
tol = 1e-6

function symmetries_from_strain(model0, voigt_strain)
    lattice = DFTK.voigt_strain_to_full(voigt_strain) * model0.lattice
    model = Model(model0; lattice, symmetries=true)
    model.symmetries
end

strain_pattern = [1., 0., 0., 1., 0., 0.];  # should yield [c11, c12, c12, c44, 0, 0] for cubic crystal

# For elastic constants beyond the bulk modulus, symmetry-breaking strains
# are required. That is, the symmetry group of the crystal is reduced.
# Here we simply precompute the relevant subgroup by applying the automatic
# symmetry detection (spglib) to the finitely perturbed crystal.
symmetries_strain = symmetries_from_strain(model0, 0.01 * strain_pattern)


function stress_from_strain(model0, voigt_strain; symmetries, Ecut, kgrid, tol)
    lattice = DFTK.voigt_strain_to_full(voigt_strain) * model0.lattice
    model = Model(model0; lattice, symmetries)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    scfres = self_consistent_field(basis; tol)
    DFTK.full_stress_to_voigt(compute_stresses_cart(scfres))
end 

stress_fn(voigt_strain) = stress_from_strain(model0, voigt_strain; symmetries=symmetries_strain, Ecut, kgrid, tol)
stress, (dstress,) = value_and_pushforward(stress_fn, AutoForwardDiff(), zeros(6), (strain_pattern,));
@show stress dstress;

c11 = uconvert(u"GPa", dstress[1] * u"hartree" / u"bohr"^3)
c12 = uconvert(u"GPa", dstress[2] * u"hartree" / u"bohr"^3)
c44 = uconvert(u"GPa", dstress[4] * u"hartree" / u"bohr"^3)
@show c11 c12 c44;

# These results can be compared directly to finite differences of the stress-strain relation:
h = 1e-3
dstress_fd = (stress_fn(h * strain_pattern) - stress_fn(-h * strain_pattern)) / 2h
c11_fd = uconvert(u"GPa", dstress_fd[1] * u"hartree" / u"bohr"^3)
c12_fd = uconvert(u"GPa", dstress_fd[2] * u"hartree" / u"bohr"^3)
c44_fd = uconvert(u"GPa", dstress_fd[4] * u"hartree" / u"bohr"^3)
@show c11_fd c12_fd c44_fd;

# Here are AD-DFPT results from increasing discretization parameters:
#
# | Ecut | kgrid         | c11    | c12   | c44    |
# |------|---------------|-------:|------:|-------:|
# | 18   | [4, 4, 4]     | 156.51 | 59.57 |  98.61 |
# | 18   | [8, 8, 8]     | 153.53 | 56.90 | 100.07 |
# | 24   | [8, 8, 8]     | 153.26 | 56.82 |  99.97 |
# | 24   | [14, 14, 14]  | 153.03 | 56.71 | 100.09 |
#
# For comparison, Materials Project for PBE *relaxed-ion* elastic constants of silicon [mp-149](https://next-gen.materialsproject.org/materials/mp-149):
# c11 = 153 GPa, c12 = 57 GPa, c44 = 74 GPa.
# Note the discrepancy in c44, which is due to us not yet including ionic relaxation in this example.

