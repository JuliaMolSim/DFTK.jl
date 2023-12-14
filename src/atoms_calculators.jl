#
# Define AtomsCalculators interface for DFTK.
#
# This interface is inspired by the one used in Molly.jl, 
# see https://github.com/JuliaMolSim/Molly.jl/blob/master/src/types.jl
#
# TODO: Decide how Hamiltonian terms should be updated upon change of the positions.
# These terms are held inside the `basis`. Currently, the whole basis is rebuilt.
# TODO: Find out if symmetries can be re-enabled (see issue on GH).

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
    basis = prepare_basis(system, params)
    ρ = guess_density(basis, system)
    ψ = nothing # Will get initialized by SCF.
    DFTKState((; ρ, ψ, basis))
end

function prepare_basis(system::AbstractSystem, params::DFTKParameters)
    # By default create an LDA model.
    model = model_DFT(system; symmetries=false, params.model_kwargs...)
    PlaneWaveBasis(model; params.basis_kwargs...)
end

Base.@kwdef mutable struct DFTKCalculator
    params::DFTKParameters
    state::DFTKState = nothing
end

function DFTKCalculator(system; state=nothing, verbose_scf=false, model_kwargs, basis_kwargs, scf_kwargs)
    if !verbose_scf
        scf_kwargs = merge(scf_kwargs, (;callback = identity))
    end
    params = DFTKParameters(;model_kwargs, basis_kwargs, scf_kwargs)

    # Create dummy state if not given.
    if isnothing(state)
        state = DFTKState(system, params)
    end
    DFTKCalculator(params, state)
end

AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(
        system::AbstractSystem, calculator::DFTKCalculator; state = calculator.state)
    # Update basis (and the enregy terms within).
    basis = prepare_basis(system, calculator.params)
    
    # Note that we use the state's densities and orbitals, but change the basis 
    # to reflect system changes.
    scfres = self_consistent_field(basis;
                    state.scfres.ρ, state.scfres.ψ, calculator.params.scf_kwargs...)
    calculator.state = DFTKState(scfres)
    calculator.state.scfres.energies.total
end
    
AtomsCalculators.@generate_interface function AtomsCalculators.forces(
        system::AbstractSystem, calculator::DFTKCalculator; state = calculator.state)
    # Update basis (and the enregy terms within).
    basis = prepare_basis(system, calculator.params)
    
    scfres = self_consistent_field(basis;
                    state.scfres.ρ, state.scfres.ψ, calculator.params.scf_kwargs...)
    calculator.state = DFTKState(scfres)
    compute_forces_cart(calculator.state.scfres)
end
