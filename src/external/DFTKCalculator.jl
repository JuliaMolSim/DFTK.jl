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

function Base.show(io::IO, calc::DFTKCalculator)
    fields = String[]
    for key in (:functionals, :pseudopotentials, :temperature, :smearing)
        if haskey(calc.params.model_kwargs, key)
            push!(fields, "$key=$(getproperty(calc.params.model_kwargs, key))")
        end
    end
    for key in (:Ecut, :kgrid)
        if haskey(calc.params.basis_kwargs, key)
            push!(fields, "$key=$(getproperty(calc.params.basis_kwargs, key))")
        end
    end
    print(io, "DFTKCalculator($(join(fields, ", ")))")
end

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

function is_converged_state(calc::DFTKCalculator, newmodel::Model, oldstate)
    newmodel != oldstate.basis.model && return false
    println("model agrees")

    oldbasis  = oldstate.basis
    newparams = calc.params.basis_kwargs
    numerical_parameters_agree = all(
        (hasproperty(oldbasis, symbol) &&
         getproperty(oldbasis, symbol) == getproperty(newparams, symbol))
        for symbol in propertynames(newparams)
    )
    !numerical_parameters_agree && return false
    println("parameters agrees")

    return calc.params.scf_kwargs.is_converged(oldstate)
end


function compute_scf(system::AbstractSystem, calc::DFTKCalculator, oldstate)
    # We re-use the symmetries from the oldstate to avoid issues if system
    # happens to be more symmetric than the structure used to make the oldstate.
    symmetries = haskey(oldstate, :basis) ? oldstate.basis.model.symmetries : true
    model = model_DFT(system; symmetries, calc.params.model_kwargs...)

    if is_converged_state(calc, model, oldstate)
        return oldstate
    end

    basis = PlaneWaveBasis(model; calc.params.basis_kwargs...)

    # @something makes sure that the density is only evaluated if ρ not in the state
    ρ = @something get(oldstate, :ρ, nothing) guess_density(basis, system)
    ψ = get(oldstate, :ψ, nothing)
    #
    #
    error("Check orbitals and density size are compatible")
    # TODO Be more clever here in particular if the lattice changes
    #      ... where right now we will get an error
    #
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
