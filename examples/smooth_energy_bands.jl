# # Ensure energy bands regularity with modified kinetic energies.
#
# TEXT

using DFTK
using Plots

# Lattice constant of silicon in bohr
a_eq = 10.26
a_list = LinRange(a_eq - 0.2, a_eq + 0.2, 2) # 50 points for the actual thing

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

#
# TEXT
#

Ecut=5
kgrid=[8,8,8]
n_bands=8
silicon = silicon_PBE(;kgrid, Ecut)

# Compute total energies for standard discretization
E_std = silicon.compute_GS.(Ref(Kinetic()), a_list; n_bands)
E_mod = silicon.compute_GS.(Ref(ModifiedKinetic(;blow_up_rate=2)), a_list; n_bands)

#
# TEXT
#

# Plot result
p = plot()
default(linewidth=1.2, framestyle=:box, grid=:true, size=(500,500))
shift = sum(abs.(E_std .- E_mod)) / length(a_list)
plot!(p, a_list, E_std, label="Standard E_0")
plot!(p, a_list, E_mod .- shift, label="Modified E_0")
xlabel!(p, "Lattice constant (bohr)")
ylabel!(p, "Total energy (hartree)")
