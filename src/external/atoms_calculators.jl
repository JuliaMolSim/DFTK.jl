# Define AtomsCalculators interface for DFTK.
#
# This interface is inspired by the one used in Molly.jl,
# see https://github.com/JuliaMolSim/Molly.jl/blob/master/src/types.jl
using AtomsBase
using AtomsCalculators


Base.@kwdef struct DFTKParameters
    model_kwargs = (; )
    basis_kwargs = (; )
    scf_kwargs = (; )
end

struct DFTKState
    scfres::NamedTuple
end
function DFTKState(system::AbstractSystem, params::DFTKParameters)
    model = model_DFT(system; params.model_kwargs...)
    basis = PlaneWaveBasis(model; params.basis_kwargs...)
    ρ = guess_density(basis, system)
    ψ = nothing  # Will get initialized by SCF.
    DFTKState((; ρ, ψ, basis))
end

Base.@kwdef mutable struct DFTKCalculator
    params::DFTKParameters
    state::DFTKState
end

"""
Construct a [AtomsCalculators](https://github.com/JuliaMolSim/AtomsCalculators.jl)
compatible calculator for DFTK. The `model_kwargs` are passed onto the
[`Model`](@ref) constructor, the `basis_kwargs` to the [`PlaneWaveBasis`](@ref)
constructor, the `scf_kwargs` to [`self_consistent_field`](@ref). At the very
least the DFT `functionals` and the `Ecut` needs to be specified.

By default the calculator preserves the symmetries that are stored inside the
`state` (the basis is re-built, but symmetries are fixed and not re-computed).

## Example
```julia-repl
julia> DFTKCalculator(system;
                      model_kwargs=(; functionals=[:lda_x, :lda_c_vwn]),
                      basis_kwargs=(; Ecut=10, kgrid=(2, 2, 2)),
                      scf_kwargs=(; tol=1e-4))
```
"""
function DFTKCalculator(system; state=nothing, verbose=false,
                        model_kwargs, basis_kwargs, scf_kwargs)
    if !verbose
        scf_kwargs = merge(scf_kwargs, (; callback=identity))
    end
    params = DFTKParameters(; model_kwargs, basis_kwargs, scf_kwargs)

    # Create dummy state if not given.
    if isnothing(state)
        state = DFTKState(system, params)
    end
    DFTKCalculator(params, state)
end

function compute_scf!(system::AbstractSystem, calculator::DFTKCalculator, state::DFTKState)
    # Note that we re-use the symmetries from the state, to avoid degenerate cases.
    model  = model_DFT(system; calculator.params.model_kwargs...,
                       symmetries=state.scfres.basis.symmetries)
    basis  = PlaneWaveBasis(model; calculator.params.basis_kwargs...)
    scfres = self_consistent_field(basis;
                                   state.scfres.ρ, state.scfres.ψ,
                                   calculator.params.scf_kwargs...)
    calculator.state = DFTKState(scfres)
end

AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(
        system::AbstractSystem, calculator::DFTKCalculator; state = calculator.state,
        kwargs...)
    compute_scf!(system, calculator, state)
    calculator.state.scfres.energies.total
end

AtomsCalculators.@generate_interface function AtomsCalculators.forces(
        system::AbstractSystem, calculator::DFTKCalculator; state = calculator.state,
        kwargs...)
    compute_scf!(system, calculator, state)
    compute_forces_cart(calculator.state.scfres)
end