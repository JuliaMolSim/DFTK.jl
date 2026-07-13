# # Geometry optimization
#
# ## Fixed cell
#
# We use DFTK and the [GeometryOptimization](https://github.com/JuliaMolSim/GeometryOptimization.jl/)
# package to find the minimal-energy bond length of the ``H_2`` molecule.
# First we set up an appropriate `DFTKCalculator` (see [AtomsCalculators integration](@ref)),
# for which we use the LDA model just like in the [Tutorial](@ref) for silicon
# in combination with a pseudodojo pseudopotential (see [Pseudopotentials](@ref)).

using DFTK
using PseudoPotentialData

pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
calc = DFTKCalculator(;
    model_kwargs = (; functionals=LDA(), pseudopotentials),  # model_DFT keyword arguments
    basis_kwargs = (; kgrid=[1, 1, 1], Ecut=10)  # PlaneWaveBasis keyword arguments
)

# Next we set up an initial hydrogen molecule within a box of vacuum.
# We use the parameters of the
# [equivalent tutorial from ABINIT](https://docs.abinit.org/tutorial/base1/)
# and DFTK's [AtomsBase integration](@ref) to setup the hydrogen molecule.
# We employ a simulation box of 10 bohr times 10 bohr times 10 bohr and a
# pseudodojo pseudopotential.
using LinearAlgebra
using Unitful
using UnitfulAtomic

r0 = 1.4   # Initial bond length in Bohr
a  = 10.0  # Box size in Bohr

cell_vectors = [[a, 0, 0]u"bohr", [0, a, 0]u"bohr", [0, 0, a]u"bohr"]
h2_crude = periodic_system([:H => [0, 0, 0.]u"bohr",
                            :H => [r0, 0, 0]u"bohr"],
                           cell_vectors)

# Finally we call `minimize_energy!` to start the geometry optimisation.
# We use `verbosity=2` to get some insight into the minimisation.
# With `verbosity=1` only a summarising table would be printed and with
# `verbosity=0` (default) the minimisation would be quiet.

using GeometryOptimization
results = minimize_energy!(h2_crude, calc; tol_forces=2e-6, verbosity=2)
nothing  # hide

# Structure after optimisation (note that the atom has wrapped around)

results.system

# Compute final bond length:

rmin = norm(position(results.system[1]) - position(results.system[2]))
println("Optimal bond length: ", rmin)

# Our result (1.486 Bohr) agrees with the
# [equivalent tutorial from ABINIT](https://docs.abinit.org/tutorial/base1/).

# ## Variable cell
# Recent versions of [GeometryOptimization](https://github.com/JuliaMolSim/GeometryOptimization.jl/)
# support cell optimization as well by passing `variablecell=true` to `minimize_energy!`.
#
# For a plane-wave code like DFTK, variable cells pose an additional challenge:
# changes to the cell's size affect the used plane waves,
# leading to discontinuities in the energy and stresses.
# This can cause the geometry optimization to struggle and/or fail to converge.
#
# A practical strategy to overcome this problem is [Energy cutoff smearing](@ref).
# As a demonstration let us find the optimal lattice constant of silicon.
#
# Like before we define a calculator, this time with a `kinetic_blowup` set
# to use energy cutoff smearing:

calc = DFTKCalculator(;
    model_kwargs = (; functionals=LDA(), pseudopotentials,
                      kinetic_blowup=BlowupCHV()),
    basis_kwargs = (; kgrid=[2, 2, 2], Ecut=10)
)

# And here is our starting silicon structure:

a = 10.0u"bohr"   # Approximate Silicon lattice constant
cell_vectors = a/2 * [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
initial_silicon = periodic_system([:Si =>  ones(3)/8,
                                   :Si => -ones(3)/8],
                                  cell_vectors;
                                  fractional=true)

# We now minimize, passing `variablecell=true`:

using GeometryOptimization
results = minimize_energy!(initial_silicon, calc; variablecell=true,
                           tol_virial=2e-6, verbosity=2)
nothing  # hide

# Structure after optimization

results.system

# Since here the cell was rescaled but its shape did not change,
# we directly extract the optimal lattice constant from one of the cell vectors:

using AtomsBase
amin = AtomsBase.cell_vectors(results.system)[1][2]*2
println("Optimal lattice constant: ", amin)

# Note that while for silicon the positions of the atoms are fixed by symmetry,
# in general a variable cell optimization will try to optimize both
# the cell and the positions of the individual atoms.
