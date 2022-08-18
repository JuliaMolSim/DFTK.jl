# # Ensure energy bands regularity with energy cutoff smearing.
# TODO::::
# The standard planewave basis depends on the domain and k point.
# The size of the basis can vary a lot and induce energy irregularities
# w.r. to kpoints of unit cell volume. DFTK features modified kinetic
# terms that allow to target wanted regularity.

using DFTK
using Plots

# Lattice constant of silicon in bohr
a_0 = 10.26
a_list = LinRange(a_0 - 0.2, a_0 + 0.2, 20)

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

# We can now compute the ground state of silicon for standard and modified
# kinetic term for each given parameter ``a``.
# We use the CHV blow-up function introduced in [REF] which ensures ``C^2``
# regularity of the energy bands. DFTK also features the blow-up function
# as implemented in Abinit [REF].

Ecut = 5
kgrid = [4,4,4]
n_bands = 8
blowup=BlowupCHV()
silicon = silicon_PBE(; kgrid, Ecut)

# Compute total energies w.r. to the lattice parameter.
E0_std = silicon.compute_GS.(Ref(Kinetic()), a_list; n_bands)
E0_mod = silicon.compute_GS.(Ref(Kinetic(; blowup)), a_list; n_bands)

# We now plot the result of the computation. The ground state energy for the
# modified kinetic term is shifted for the legibility of the plot.

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
