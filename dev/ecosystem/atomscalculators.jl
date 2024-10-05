# # AtomsCalculators integration
#
# [AtomsCalculators.jl](https://github.com/JuliaMolSim/AtomsCalculators.jl) is an interface
# for doing standard computations (energies, forces, stresses, hessians) on atomistic
# structures. It is very much inspired by the calculator objects in the
# [atomistic simulation environment](https://wiki.fysik.dtu.dk/ase/index.html).
#
# DFTK by default ships a datastructure called a [`DFTKCalculator`](@ref),
# which implements the AtomsCalculators interface. A `DFTKCalculator`
# can be constructed by passing three different named tuples,
# the `model_kwargs`, the `basis_kwargs` and the `scf_kwargs`.
# The first two named tuples are passed as keyword arguments when constructing
# the DFT model using [`model_DFT`](@ref) and its discretization using
# the [`PlaneWaveBasis`](@ref). The last one is used as keyword arguments
# when running the
# [`self_consistent_field`](@ref) function on the resulting basis to solve the problem
# numerically. Thus when using the `DFTKCalculator` the user is expected to
# pass these objects exactly the keyword argument one would pass when constructing
# a `model` and `basis` and when calling `self_consistent_field`.
#
# For example, to perform the calculation of the [Tutorial](@ref) using
# the AtomsCalculators interface we define the calculator as such:

using DFTK

model_kwargs = (; functionals=LDA())
basis_kwargs = (; kgrid=[4, 4, 4], Ecut=7)
scf_kwargs   = (; tol=1e-5)
calc = DFTKCalculator(; model_kwargs, basis_kwargs, scf_kwargs)

# Note, that the `scf_kwargs` is optional and can be missing
# (then the defaults of `self_consistent_field` are used).
#
# !!! tip "Kpoints from kpoint density"
#     Note that DFTK's [`kgrid_from_maximal_spacing`](@ref) function can also be used with
#     `AbstractSystem` objects to determine an appropriate `kgrid` paramter for the `basis_kwargs`.
#     E.g. `kgrid_from_maximal_spacing(system, 0.25u"1/Å")` gives a k-point spacing of
#     `0.25` per Angström for the passed system.
#
# Based on this `calc` object we can perform a DFT calculation on bulk silicon
# according to the
# [`AtomsCalculators` interface](https://juliamolsim.github.io/AtomsCalculators.jl/stable/interface/),
# e.g.

using AtomsBuilder
using AtomsCalculators
AC = AtomsCalculators

## Bulk silicon system of the Tutorial
silicon = attach_psp(bulk(:Si); Si="hgh/lda/si-q4")
AC.potential_energy(silicon, calc)  # Compute total energy

# or we can compute the energy and forces:

results = AC.calculate((AC.Energy(), AC.Forces()), silicon, calc)
results.energy
#-
results.forces

# Note that the `results` object returned by the call to `AtomsCalculators.calculate`
# also contains a `state`, which is a DFTK `scfres`. This can be used to speed up
# subsequent computations:

## This is basically for free, since already computed:
results2 = @time AC.calculate((AC.Energy(), AC.Forces()), silicon, calc, nothing, results.state);

# For an example using the `DFTKCalculator`, see [Geometry optimization](@ref).
