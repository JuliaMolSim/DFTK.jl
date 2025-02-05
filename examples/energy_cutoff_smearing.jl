# # Energy cutoff smearing
#
# A technique that has been employed in the literature to ensure smooth energy bands
# for finite Ecut values is energy cutoff smearing.
#
# As recalled in the
# [Problems and plane-wave discretization](https://docs.dftk.org/stable/guide/periodic_problems/)
# section, the energy of periodic systems is computed by solving eigenvalue
# problems of the form
# ```math
# H_k u_k = ε_k u_k,
# ```
# for each ``k``-point in the first Brillouin zone of the system.
# Each of these eigenvalue problem is discretized with a plane-wave basis
# ``\mathcal{B}_k^{E_c}=\{x ↦ e^{iG · x} \;\;|\;G ∈ \mathcal{R}^*,\;\; |k+G|^2 ≤ 2E_c\}``
# whose size highly depends on the choice of ``k``-point, cell size or
# cutoff energy ``\rm E_c`` (the `Ecut` parameter of DFTK).
# As a result, energy bands computed along a ``k``-path in the Brillouin zone
# or with respect to the system's unit cell volume - in the case of geometry optimization
# for example - display big irregularities when `Ecut` is taken too small.
#
# Here is for example the variation of the ground state energy of diamond silicon
# with respect to its lattice parameter, around the experimental lattice constant.

using AtomsBuilder
using DFTK
using PseudoPotentialData
using Statistics
using Unitful
using UnitfulAtomic

a0 = 10.26u"bohr"  # Experimental lattice constant of silicon
a_list = a0 * range(0.98, 1.02, length=20)

function compute_ground_state_energy(a; Ecut, kgrid, kinetic_blowup, kwargs...)
    pseudopotentials = PseudoFamily("cp2k.nc.sr.pbe.v0_1.semicore.gth")
    model = model_DFT(bulk(:Si; a); functionals=PBE(), kinetic_blowup, pseudopotentials)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    self_consistent_field(basis; callback=identity, kwargs...).energies.total
end

Ecut  = 5          # Very low Ecut to display big irregularities
kgrid = (4, 4, 4)  # Very sparse k-grid to speed up convergence
E0_naive = compute_ground_state_energy.(a_list; kinetic_blowup=BlowupIdentity(), Ecut, kgrid);

# To be compared with the same computation for a high `Ecut=100`. The naive approximation
# of the energy is shifted for the legibility of the plot.
E0_ref - [-7.867399262300442, -7.867875504884598, -7.86831005961699,
          -7.868703712435519, -7.869057235894591, -7.869371393835255,
          -7.869646937992838, -7.869884611383302, -7.870085144568824,
          -7.8702492593753135, -7.870377668337734, -7.870471072484817,
          -7.870530163168231, -7.870555622590523, -7.870548125081617,
          -7.870508333380078, -7.8704369010295006, -7.870334474350743,
          -7.870201688247285, -7.870039170777946]

using Plots
shift = mean(E0_naive - E0_ref)
p = plot(a_list, E0_naive .- shift, label="Ecut=5", xlabel="lattice parameter a",
         ylabel="Ground state energy (Ha)", color=1, legend=:topright)
plot!(p, a_list, E0_ref, label="Ecut=100", color=2)

# The problem of non-smoothness of the approximated energy is typically avoided by
# taking a large enough `Ecut`, at the cost of a high computation time.
# Another method consist in introducing a modified kinetic term defined through
# the data of a blow-up function, a method which is also referred to as "energy cutoff
# smearing". DFTK features energy cutoff smearing using the CHV blow-up
# function introduced in [^CHV2024] that is mathematically ensured to provide ``C^2``
# regularity of the energy bands.
#
# [^CHV2024]:  
#    Eric Cancès, Muhammad Hassan and Laurent Vidal;
#    *Modified-operator method for the calculation of band diagrams of
#    crystalline materials*.
#    Math. Comp. 93 (2024), 1203-1245
#    DOI: https://doi.org/10.1090/mcom/3897

# Let us launch the computation again with the modified kinetic term.
E0_modified = compute_ground_state_energy.(a_list; kinetic_blowup=BlowupCHV(), Ecut, kgrid);

# !!! note "Abinit energy cutoff smearing option"
#     For the sake of completeness, DFTK also provides the blow-up function `BlowupAbinit`
#     proposed in the Abinit quantum chemistry code. This function depends on a parameter
#     `Ecutsm` fixed by the user
#     (see [Abinit user guide](https://docs.abinit.org/variables/rlx/#ecutsm)).
#     For the right choice of `Ecutsm`, `BlowupAbinit` corresponds to the `BlowupCHV` approach
#     with coefficients ensuring ``C^1`` regularity. To choose `BlowupAbinit`, pass
#     `kinetic_blowup=BlowupAbinit(Ecutsm)` to the model constructors.
#

# We can now compare the approximation of the energy as well as the estimated
# lattice constant for each strategy.

estimate_a0(E0_values) = a_list[findmin(E0_values)[2]]
a0_naive, a0_ref, a0_modified = estimate_a0.([E0_naive, E0_ref, E0_modified])

shift = mean(E0_modified - E0_ref)  # Shift for legibility of the plot
plot!(p, a_list, E0_modified .- shift, label="Ecut=5 + BlowupCHV", color=3)
vline!(p, [a0], label="experimental a0", linestyle=:dash, linecolor=:black)
vline!(p, [a0_naive], label="a0 Ecut=5", linestyle=:dash, color=1)
vline!(p, [a0_ref], label="a0 Ecut=100", linestyle=:dash, color=2)
vline!(p, [a0_modified], label="a0 Ecut=5 + BlowupCHV", linestyle=:dash, color=3)

# The smoothed curve obtained with the modified kinetic term allow to clearly designate
# a minimal value of the energy with respect to the lattice parameter ``a``, even with
# the low `Ecut=5` Ha.

println("Error of approximation of the reference a0 with modified kinetic term:"*
        " $(round((a0_modified - a0_ref)*100/a0_ref, digits=5))%")
