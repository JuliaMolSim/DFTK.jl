#
# Define AtomsCalculators interface for DFTK.
#
# This interface is inspired by the one used in Molly.jl, 
# see https://github.com/JuliaMolSim/Molly.jl/blob/master/src/types.jl
#
# By default, when the calculator is called with a `state`, the 
# symmetries of the state will be re-used in the current calculation (the basis 
# is re-built, but symmetries are fixed and not re-computed). 
#    
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
    ψ = nothing # Will get initialized by SCF.
    DFTKState((; ρ, ψ, basis))
end

Base.@kwdef mutable struct DFTKCalculator
    params::DFTKParameters
    state::DFTKState
end

function DFTKCalculator(system; state=nothing, verbose=false,
                        model_kwargs, basis_kwargs, scf_kwargs)
    if !verbose
        scf_kwargs = merge(scf_kwargs, (;callback=identity))
    end
    params = DFTKParameters(;model_kwargs, basis_kwargs, scf_kwargs)

    # Create dummy state if not given.
    if isnothing(state)
        state = DFTKState(system, params)
    end
    DFTKCalculator(params, state)
end

function compute_scf!(system::AbstractSystem, calculator::DFTKCalculator, state::DFTKState)
    # Note that we re-use the symmetries from the state, to avoid degenerate cases.
    model = model_DFT(system; calculator.params.model_kwargs...,
                      symmetries=state.scfres.basis.symmetries)
    basis = PlaneWaveBasis(model; calculator.params.basis_kwargs...)
    
    # Note that we use the state's densities and orbitals, but change the basis 
    # to reflect system changes.
    scfres = self_consistent_field(basis;
                                   state.scfres.ρ, state.scfres.ψ,
                                   calculator.params.scf_kwargs...)
    calculator.state = DFTKState(scfres)
end

AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(
        system::AbstractSystem, calculator::DFTKCalculator; state = calculator.state)
    compute_scf!(system, calculator, state)
    calculator.state.scfres.energies.total
end
    
AtomsCalculators.@generate_interface function AtomsCalculators.forces(
        system::AbstractSystem, calculator::DFTKCalculator; state = calculator.state)
    compute_scf!(system, calculator, state)
    compute_forces_cart(calculator.state.scfres)
end
