# # Ensure energy bands regularity with energy cutoff smearing.
#
# For a given system, the size of the standard ``k``-point dependant
# plane-wave discretization basis is highly dependant of the chosen ``k``-point, cut-off
# energy ``E_c`` as well as the size of the system's unit cell. As a result, energy bands
# computed along a ``k``-path in the Brillouin zone, or with respect to the system's
# unit cell volume - in the case of geometry optimization for example -
# display big irregularities when ``E_c`` is taken to small. The problem can be tackled by
# introducing  a modified kinetic term in the Hamiltonian.
# 
# A modified kinetic term is implemeneted in DFTK, that is mathematicaly ensured to provide
# C^2 regularity of the energy bands. Let us give a brief example of the usage of such a modified
# term in the case of the numerical estimation of the lattice constant of face centered
# crytaline (FCC) silicon.

using DFTK
using Plots

# The lattice of FCC silicon only depends on a single parameter ``a``. We want to
# plot the variation of the ground state energy with respect to ``a`` to estimate the
# constant of minimal energy ``a_0``. For this example, let us centered the plot
# around the experimental value of ``a_0``.

a_0 = 10.26 # Experimental lattice constant of silicon in bohr

# TODO: I put only two points here because the full computation for 100 points
# is a bit heavy and I didn't managed to directly put the plot as a pdf file.
# I have to do that in the next commit..
a_list = LinRange(a_0 - 0.2, a_0 + 0.2, 2)

# In order to easily compare the performances of standard and modified terms,
# we define the silicon model and plane-wave basis direcly as functions of the
# kinetic term and lattice parameter ``a``.

PBE_terms(KineticTerm) = [KineticTerm, AtomicLocal(), AtomicNonlocal(),
        Ewald(), PspCorrection(), Hartree(), Xc([:gga_x_pbe, :gga_c_pbe])]

function silicon_PBE(; basis_kwargs...)
    # Defines a model for silicon with PBE functional given kinetic term and
    # lattice parameter a
    function model_silicon(KineticTerm, a)
        lattice = a / 2 * [[0 1 1.];
                           [1 0 1.];
                           [1 1 0.]]
        Si = ElementPsp(:Si, psp=load_psp("hgh/pbe/Si-q4"))
        atoms = [Si, Si]
        positions = [ones(3)/8, -ones(3)/8]
        Model(lattice, atoms, positions; terms=PBE_terms(KineticTerm),
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
# We use the CHV blow-up function introduced in [REF] which ensures ``C^2``
# regularity of the energy bands.

Ecut = 5 # very low Ecut to display big irregularities
kgrid = [4,4,4]
n_bands = 8
blowup=BlowupCHV()
silicon = silicon_PBE(; kgrid, Ecut)

# Compute total energies w.r. to the lattice parameter.
E0_std = silicon.compute_GS.(Ref(Kinetic()), a_list; n_bands)
E0_mod = silicon.compute_GS.(Ref(Kinetic(; blowup)), a_list; n_bands)

# Let us plot the result of the computation. The ground state energy for the
# modified kinetic term is shifted for the legibility of the plot.
# TODO: replace by a precomputed pdf file.
p = plot()
default(linewidth=1.2, framestyle=:box, grid=:true, size=(500,500))
shift = sum(abs.(E0_std .- E0_mod)) / length(a_list)
plot!(p, a_list, E0_std, label="Standard E_0")
plot!(p, a_list, E0_mod .- shift, label="Modified shifted E_0")
xlabel!(p, "Lattice constant (bohr)")
ylabel!(p, "Total energy (hartree)")

# The smoothed curve allow to estimate the equilibrium lattice constant
# around 10.31 bohr, which corresponds, up to the error of the PBE approximation,
# to the experimental value of 10.26 bohr. [Do I add the ref for the PBE error ?]
