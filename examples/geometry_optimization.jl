# # Geometry optimization
#
# We use DFTK and the [GeometryOptimization](https://github.com/JuliaMolSim/GeometryOptimization.jl/)
# package to find the minimal-energy bond length of the ``H_2`` molecule.
# First we set up an appropriate `DFTKCalculator` (see [AtomsCalculators integration](@ref)),
# for which we use the LDA model just like in the [Tutorial](@ref) for silicon.

calc = DFTKCalculator(;
    model_kwargs = (; functionals=[:lda_x, :lda_c_pw]),  # xc functionals employed by model_LDA
    basis_kwargs = (; kgrid=[1, 1, 1], Ecut=10)          # Crude numerical parameters
)

# Next we set up an initial hydrogen molecule within a box of vacuum:
r0 = 1.4   # Initial bond length in Bohr
a  = 10.0  # Box size in Bohr

lattice = a * I(3)
H = ElementPsp(:H; psp=load_psp("hgh/lda/h-q1"));
atoms = [H, H]
positions = [zeros(3), lattice \ [r0, 0., 0.]]

h2_crude = periodic_system(lattice, atoms, positions)

# Finally we call `minimize_energy!` to start the geometry optimisation:

using GeometryOptimization
results = minimize_energy!(h2_crude, calc; tol_force=2e-6)
results.system  # Print final system

# Compute final bond length:

rmin = norm(position(results.system[1]) - position(results.system[2]))
@printf "\nOptimal bond length for Ecut=%.2f: %.3f Bohr\n" Ecut austrip(rmin)

# Our results (1.523 Bohr) agrees agrees with the
# [equivalent tutorial from ABINIT](https://docs.abinit.org/tutorial/base1/).
