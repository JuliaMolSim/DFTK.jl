# Define AtomsCalculators interface for DFTK.
using AtomsBase
import AtomsCalculators
import AtomsCalculators: @generate_interface


Base.@kwdef struct DFTKParameters
    model_kwargs = (; )
    basis_kwargs = (; )
    scf_kwargs   = (; )
end

struct DFTKCalculator{T}
    # Note: The params are *not all* parameters in the sense of the LUX interface convention,
    #       hence we do not expose them with `AtomsCalculators.get_parameters`
    params::DFTKParameters
    st::T
    #
    # Calculator counters
    # TODO The Ref thingies feel a little wrong, somehow this should be part of the
    #      state, but this may make it hard to keep track during geometry optimisation
    #      or similar. In any case don't rely on this for now, it may disappear.
    counter_n_iter::Ref{Int}
    counter_matvec::Ref{Int}
    #
    # Calculator parameters
    enforce_convergence::Bool  # If true, throws an error exception on non-convergence

    function DFTKCalculator(params::DFTKParameters, st=nothing; enforce_convergence=true)
        new{Nothing}(params, st, Ref(0), Ref(0), enforce_convergence)
    end
end
AtomsCalculators.energy_unit(::DFTKCalculator) = u"hartree"
AtomsCalculators.length_unit(::DFTKCalculator) = u"bohr"

# TODO DFTKCalculator needs a custom show function (both brief and detailed)

"""
Construct a [AtomsCalculators](https://github.com/JuliaMolSim/AtomsCalculators.jl)
compatible calculator for DFTK. The `model_kwargs` are passed onto the
[`Model`](@ref) constructor, the `basis_kwargs` to the [`PlaneWaveBasis`](@ref)
constructor, the `scf_kwargs` to [`self_consistent_field`](@ref). At the very
least the DFT `functionals` and the `Ecut` needs to be specified.

By default the calculator preserves the symmetries that are stored inside the
`st` (the basis is re-built, but symmetries are fixed and not re-computed).

Calculator-specific keyword arguments are:
- `verbose`: If true, the SCF iterations are printed.
- `enforce_convergence`: If false, the calculator does not error out
  in case of a non-converging SCF.

## Example
```julia-repl
julia> DFTKCalculator(; model_kwargs=(; functionals=LDA()),
                        basis_kwargs=(; Ecut=10, kgrid=(2, 2, 2)),
                        scf_kwargs=(; tol=1e-4))
```
"""
function DFTKCalculator(; verbose=false, model_kwargs, basis_kwargs, scf_kwargs=(; ),
                          st=nothing, kwargs...)
    if !verbose && !(:callback in keys(scf_kwargs))
        # If callback is given in scf_kwargs, then this automatically overwrites
        # the default callback, which prints the iterations.
        scf_kwargs = merge(scf_kwargs, (; callback=identity))
    end
    DFTKCalculator(DFTKParameters(; model_kwargs, basis_kwargs, scf_kwargs), st; kwargs...)
end

# TODO Do something with parameters ?
AtomsCalculators.get_state(calc::DFTKCalculator) = calc.st
function AtomsCalculators.set_state!(calc::DFTKCalculator, st)
    DFTKCalculator(calc.params, st; calc.enforce_convergence)
end


function compute_scf(system::AbstractSystem, calc::DFTKCalculator, oldstate)
    # We re-use the symmetries from the oldstate to avoid issues if system
    # happens to be more symmetric than the structure used to make the oldstate.
    symmetries = haskey(oldstate, :basis) ? oldstate.basis.model.symmetries : true
    model = model_DFT(system; symmetries, calc.params.model_kwargs...)
    basis = PlaneWaveBasis(model; calc.params.basis_kwargs...)

    # @something makes sure that the density is only evaluated if ρ not in the state
    ρ = @something get(oldstate, :ρ, nothing) guess_density(basis, system)
    ψ = get(oldstate, :ψ, nothing)
    scfres = self_consistent_field(basis; ρ, ψ, calc.params.scf_kwargs...)
    calc.enforce_convergence && !scfres.converged && error("SCF not converged.")
    calc.counter_n_iter[] += scfres.n_iter
    calc.counter_matvec[] += scfres.n_matvec
    scfres
end
function compute_scf(system::AbstractSystem, calc::DFTKCalculator, ::Nothing)
    compute_scf(system, calc, (; ))
end


@generate_interface function AtomsCalculators.calculate(::AtomsCalculators.Energy,
        system::AbstractSystem, calc::DFTKCalculator, ps=nothing, st=nothing; kwargs...)
    scfres = compute_scf(system, calc, st)
    (; energy=scfres.energies.total * u"hartree", state=scfres)
end

@generate_interface function AtomsCalculators.calculate(::AtomsCalculators.Forces,
        system::AbstractSystem, calc::DFTKCalculator, ps=nothing, st=nothing; kwargs...)
    scfres = compute_scf(system, calc, st)
    (; forces=compute_forces_cart(scfres) * u"hartree/bohr",
       energy=scfres.energies.total * u"hartree",
       state=scfres)
end

@generate_interface function AtomsCalculators.calculate(::AtomsCalculators.Virial,
        system::AbstractSystem, calc::DFTKCalculator, ps=nothing, st=nothing; kwargs...)
    scfres  = compute_scf(system, calc, st)
    Ω = scfres.basis.model.unit_cell_volume
    virial = (-Ω * compute_stresses_cart(scfres)) * u"hartree"
    (; virial, energy=scfres.energies.total * u"hartree", state=scfres)
end


# TODO Something more clever when energy + other stuff is needed
#      - This is right now tricky in AtomsCalculators, since energy_forces for example
#        dispatches to potential_energy and forces, which is not able to make
#        use of state sharing.
