# Define AtomsCalculators interface for DFTK.
#
# This interface is inspired by the one used in Molly.jl,
# see https://github.com/JuliaMolSim/Molly.jl/blob/master/src/types.jl
using AtomsBase
using AtomsCalculators


Base.@kwdef struct DFTKParameters
    model_kwargs = (; )
    basis_kwargs = (; )
    scf_kwargs   = (; )
end

struct DFTKState{T}
    scfres::T
end
DFTKState() = DFTKState((; ψ=nothing, ρ=nothing))

mutable struct DFTKCalculator
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
julia> DFTKCalculator(; model_kwargs=(; functionals=[:lda_x, :lda_c_vwn]),
                        basis_kwargs=(; Ecut=10, kgrid=(2, 2, 2)),
                        scf_kwargs=(; tol=1e-4))
```
"""
function DFTKCalculator(params::DFTKParameters)
    DFTKCalculator(params, DFTKState())  # Create dummy state if not given.
end

function DFTKCalculator(; verbose=false, model_kwargs, basis_kwargs, scf_kwargs)
    if !verbose
        scf_kwargs = merge(scf_kwargs, (; callback=identity))
    end
    params = DFTKParameters(; model_kwargs, basis_kwargs, scf_kwargs)
    DFTKCalculator(params)
end

function compute_scf!(system::AbstractSystem, calculator::DFTKCalculator, state::DFTKState)
    params = calculator.params

    # We re-use the symmetries from the state to avoid issues
    # with accidentally more symmetric structures.
    symmetries = haskey(state.scfres, :basis) ? state.scfres.basis.model.symmetries : true
    model = model_DFT(system; symmetries, params.model_kwargs...)
    basis = PlaneWaveBasis(model; params.basis_kwargs...)

    ρ = @something state.scfres.ρ guess_density(basis, system)
    scfres = self_consistent_field(basis; ρ, state.scfres.ψ, params.scf_kwargs...)
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
