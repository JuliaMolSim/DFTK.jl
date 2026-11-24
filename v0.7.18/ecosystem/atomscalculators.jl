# # AtomsCalculators integration
#
# [AtomsCalculators.jl](https://github.com/JuliaMolSim/AtomsCalculators.jl) is an interface
# for doing standard computations (energies, forces, stresses, hessians) on atomistic
# structures. It is very much inspired by the calculator objects in the
# [atomistic simulation environment](https://ase-lib.org/).
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
using PseudoPotentialData

pd_lda_family = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
model_kwargs  = (; functionals=LDA(), pseudopotentials=pd_lda_family)
basis_kwargs  = (; kgrid=KgridSpacing(0.3), Ecut=7)
scf_kwargs    = (; tol=1e-5)
calc = DFTKCalculator(; model_kwargs, basis_kwargs, scf_kwargs)

# Note, that the `scf_kwargs` is optional and can be missing
# (then the defaults of `self_consistent_field` are used).
#
# !!! tip "Kpoints from kpoint density"
#     Note that as the `kgrid` parameter as part of the `basis_kwargs`
#     (the argument passed to the construction of the [`PlaneWaveBasis`](@ref))
#     you can also pass a [`KgridSpacing`](@ref) and a [`KgridTotalNumber`](@ref)
#     object, which determines the actual number of k-point to use from the
#     `AbstractSystem` on which the calculation is performed.
#     E.g. `kgrid=KgridSpacing(0.25u"1/Å")` yields a k-point spacing of
#     `0.25` per Angström for the system on which the calculation is performed.
#
# Based on this `calc` object we can perform a DFT calculation on bulk silicon
# according to the
# [`AtomsCalculators` interface](https://juliamolsim.github.io/AtomsCalculators.jl/stable/interface/),
# e.g.

using AtomsBuilder
using AtomsCalculators
AC = AtomsCalculators

## Bulk silicon system of the Tutorial
silicon = bulk(:Si)
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
