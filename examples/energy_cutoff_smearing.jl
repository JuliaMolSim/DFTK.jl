# # Ensure energy bands regularity with energy cutoff smearing.
#
# As recalled in the
# [Problems and plane-wave discretization](https://docs.dftk.org/stable/guide/periodic_problems/)
# section, the energy of periodic systems is computed by solving eigenvalue
# problems of the form
#
#```math
# \begin{equation}
# H_ku_k=\varepsilon_k u_k
# \end{equation}
# ```
#
# for each ``k``-point in the first Brillouin zone of the system.
# Each of these eigenvalue problem is discretized with a plane-wave basis
# ``\mathcal{B}_k^{E_c}=\{x\mapsto e^{iG\cdot x} \;\;|\;G\in\mathcal{R}^*,\;\; |k+G|^2\leq 2E_c\}``
# whose size highly depends on the choice of ``k``-point, cutoff energy ``\rm E_c`` or
# cell size.  As a result, energy bands computed along a ``k``-path in the Brillouin zone
# or with respect to the system's unit cell volume - in the case of geometry optimization
# for example - display big irregularities when ``E_c`` is taken too small.
#
# Here is for example the variation of the ground state energy of face cubic centered
# (FCC) silicon with respect to its lattice parameter,
# around the experimental lattice constant.

using DFTK

a0 = 10.26 # Experimental lattice constant of silicon in bohr
a_list = LinRange(a0 - 1/2, a0 + 1/2, 20) # 20 points around a0

Ecut = 5        # very low Ecut to display big irregularities
kgrid = [2,2,2] # very sparse k-grid to fasten convergence
n_bands = 8     # Standard number of bands for silicon

function compute_ground_state_energy(a; Ecut, kgrid, kwargs...)
    function model(a)
        lattice = a / 2 * [[0 1 1.];
                           [1 0 1.];
                           [1 1 0.]]
        Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
        atoms = [Si, Si]
        positions = [ones(3)/8, -ones(3)/8]
        model_PBE(lattice, atoms, positions)
    end
    basis(a) = PlaneWaveBasis(model(a); Ecut, kgrid)
    self_consistent_field(basis(a); kwargs...).energies.total
end

callback = info->nothing # set SCF to non verbose
E0_naive = compute_ground_state_energy.(a_list; Ecut, kgrid, n_bands, callback);

# To be compared with the same computation for a high `Ecut=100`. The naive approximation
# of the energy is shifted for the legibility of the plot.
E0_ref = [-7.839775223322127, -7.843031658146996, -7.845961005280923,
          -7.848576991754026, -7.850892888614151, -7.852921532056932,
          -7.854675317792186, -7.85616622262217,  -7.85740584131599,
          -7.858405359984107, -7.859175611288143, -7.859727053496513,
          -7.860069804791132, -7.860213631865354, -7.8601679947736915,
          -7.859942011410533, -7.859544518721661, -7.858984032385052,
          -7.858268793303855, -7.857406769423708]

using Plots
shift = sum(abs.(E0_naive .- E0_ref)) / 20
p = plot(a_list, E0_naive .- shift, label="Ecut=5 Ha", xlabel="lattice parameter a (bohr)",
         ylabel="Ground state energy (Ha)")
plot!(p, a_list, E0_ref, label="Ecut=100 Ha")

# The problem of non-smoothness of the approximated energy is typically avoided by
# taking a large enough `Ecut`, at the cost of a high computation time.
# Another method consist in introducing a modified kinetic term defined through
# the data of a blow-up funtion, a method which is also refered to as "energy cutoff
# smearing". DFTK features energy cutoff smearing using the CHV blow-up
# function introduced in [REF of the paper to be submitted],
# that is mathematicaly ensured to provide C^2 regularity of the energy bands.

# !!! note "Other energy cutoff options"
#     The quantum chemistry codes Qbox [^Qbox] and Abinit [^Abinit] also feature
#     energy cutoff smearing options. The Abinit blow-up function corresponds to
#     the CHV one, with a choice of coefficients ensuring ``C^1`` regularity. The Qbox
#     implementation however doesn't use a proper blow-up function which in turn has no
#     mathematical proof of regularization of the energy bands.
#

# [^Qbox]:
#    Qbox first principles molecular dynamics
#    [documentation](http://qboxcode.org/doc/html/usage/variables.html#ecuts-var)
# [^Abinit]:
#    Abinit software suite [user guide](https://docs.abinit.org/variables/rlx/#ecutsm)
#

# Let us lauch the computation again with the modified kinetic term.

blowup = BlowupCHV() # Choose blowup function
modified_PBE_terms = [Kinetic(;blowup), AtomicLocal(), AtomicNonlocal(),
                      Ewald(), PspCorrection(), Hartree(), Xc([:gga_x_pbe, :gga_c_pbe])]

function compute_ground_state_energy_modified(a; Ecut, kgrid, kwargs...)
    function model(a)
        lattice = a / 2 * [[0 1 1.];
                           [1 0 1.];
                           [1 1 0.]]
        Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
        atoms = [Si, Si]
        positions = [ones(3)/8, -ones(3)/8]
        Model(lattice, atoms, positions; terms=modified_PBE_terms)
    end
    basis(a) = PlaneWaveBasis(model(a); Ecut, kgrid)
    self_consistent_field(basis(a); kwargs...).energies.total
end

E0_modified = compute_ground_state_energy_modified.(a_list; Ecut, kgrid,
                                                    n_bands, callback);

# We can know compare the approximation of the energy as well as the estimated
# lattice constant for each strategy.

estimate_a0(E0_values) = a_list[findmin(E0_values)[2]]
a0_naive, a0_ref, a0_modified = estimate_a0.([E0_naive, E0_ref, E0_modified])

shift = sum(abs.(E0_modified .- E0_ref)) / 20 # again, shift for legibility of the plot

plot!(p, a_list, E0_modified .- shift, label="Ecut=5 Ha | modified kinetic term")
vline!(p, [a0], label="experimental a0", linestyle=:dash, linecolor=:black)
vline!(p, [a0_naive], label="a0 Ecut=5", linestyle=:dash)
vline!(p, [a0_ref], label="a0 Ecut=100", linestyle=:dash)
vline!(p, [a0_modified], label="a0 Ecut=5 | modified kinetic term", linestyle=:dash)

# The smoothed curve obtained with the modified kinetic term allow to clearly designate
# a minimal value of the energy with respect to the lattice parameter ``a``, even with
# the low `Ecut=5` Ha. It matches the approximation of the lattice constant obtained
# with `Ecut=100` Ha within an error of 0.5%.

println("Error of approximation of the reference a0 with modified kinetic term:"*
        " $((a0_modified - a0_ref)*100/a0_ref)")
