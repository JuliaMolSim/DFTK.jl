# # Ensure energy bands regularity with energy cutoff smearing.
#
# For a given system, the size of the standard ``k``-point dependant
# plane-wave discretization basis is highly dependant of the chosen ``k``-point, cut-off
# energy ``E_c`` as well as the size of the system's unit cell. As a result, energy bands
# computed along a ``k``-path in the Brillouin zone, or with respect to the system's
# unit cell volume - in the case of geometry optimization for example -
# display big irregularities when ``E_c`` is taken to small. The problem can be tackled by
# introducing a modified kinetic term in the Hamiltonian.
# 
# A modified kinetic term is implemeneted in DFTK, that is mathematicaly
# ensured to provide C^2 regularity of the energy bands.
# Let us give a brief example of the usage of such a modified term in the case
# of the numerical estimation of the lattice constant of face centered
# crytaline (FCC) silicon.

using DFTK
using Plots

# The lattice of FCC silicon only depends on a single parameter ``a``. We want to
# plot the variation of the ground state energy with respect to ``a`` to estimate the
# constant of minimal energy ``a_0``. For this example, let us centered the plot
# around the experimental value of ``a_0``.

a_0 = 10.26 # Experimental lattice constant of silicon in bohr
a_list = LinRange(a_0 - 1/2, a_0 + 1/2, 20) # 20 points around a0

# In order to easily compare the performances of standard and modified terms,
# we define the silicon model and plane-wave basis direcly as functions of the
# kinetic term and lattice parameter ``a``.

LDA_terms(KineticTerm) = [KineticTerm, AtomicLocal(), AtomicNonlocal(),
        Ewald(), PspCorrection(), Hartree(), Xc([:lda_x, :lda_c_pw])]

function silicon_LDA(; basis_kwargs...)
    # Defines a model for silicon with LDA functional given kinetic term and
    # lattice parameter a
    function model_silicon(KineticTerm, a)
        lattice = a / 2 * [[0 1 1.];
                           [1 0 1.];
                           [1 1 0.]]
        Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
        atoms = [Si, Si]
        positions = [ones(3)/8, -ones(3)/8]
        Model(lattice, atoms, positions; terms=LDA_terms(KineticTerm),
              model_name="custom")
    end
    # Construct a plane-wave basis given a kinetic term using model_silicon
    basis_silicon(KineticTerm, a; kwargs...) =
        PlaneWaveBasis(model_silicon(KineticTerm, a); kwargs...)
    # Compute the ground state given the kinetic term and lattice parameter a
    compute_GS(KineticTerm, a; kwargs...) =
        self_consistent_field(basis_silicon(KineticTerm, a; basis_kwargs...);
                              kwargs...).energies.total
    (;compute_GS=compute_GS, basis=basis_silicon, model=model_silicon)
end

# We can now compute the wanted energies for standard and modified kinetic term.
# The modified energies are defined through the data of a blow-up function.
# We use the CHV blow-up function introduced in [REF] which coefficients chosen
# to ensures ``C^2`` regularity of the energy bands. More details on properties
# of the blow-up function can be found in the same reference.

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

# Let us set up the parameters of the SCF cycles.

Ecut = 5        # very low Ecut to display big irregularities
kgrid = [2,2,2] # very sparse k-grid to fasten convergence
n_bands = 8
blowup=BlowupCHV()
silicon = silicon_LDA(; kgrid, Ecut)

# We now compute total energies with respect to the lattice parameter...

E0_std = silicon.compute_GS.(Ref(Kinetic()), a_list; n_bands)
E0_mod = silicon.compute_GS.(Ref(Kinetic(; blowup)), a_list; n_bands)

# ... and plot the result of the computations. The ground state energy for the
# modified kinetic term is shifted for the legibility of the plot.

p = plot()
default(linewidth=1.2, framestyle=:box, grid=:true, size=(600,400))
shift = sum(abs.(E0_std .- E0_mod)) / length(a_list)
plot!(p, a_list, E0_std, label="Standard E_0")
plot!(p, a_list, E0_mod .- shift, label="Modified shifted E_0")
xlabel!(p, "Lattice constant (bohr)")
ylabel!(p, "Total energy (hartree)")

# The smoothed curve allow to clearly designate a minimal value of the energy with
# respect to ``a``. Note that this estimate still suffers from errors relative to
# the LDA approximation and the choice of sparse k-grid, for which one benefits however
# from error estimates.
