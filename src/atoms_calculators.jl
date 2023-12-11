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

struct DFTKParameters
    Ecut::Real
    kgrid::Union{Nothing,<:AbstractVector{Int}}
    tol::Real
    temperature::Real
end

struct DFTKState
    scfres::NamedTuple
end
function DFTKState(system::AbstractSystem, params::DFTKParameters)
    basis = prepare_basis(system, params)
    ρ = guess_density(basis)
    ψ = nothing # Will get initialized by SCF.
    return DFTKState((;ρ, ψ, basis))
end


mutable struct DFTKCalculator
    params::DFTKParameters
    scf_callback
    state::DFTKState
end

function prepare_basis(system::AbstractSystem, params::DFTKParameters)
    # By default create an LDA model.
    model = model_LDA(system; temperature=params.temperature, symmetries=false)
    basis = PlaneWaveBasis(model; params.Ecut, params.kgrid)
    return basis
end

function DFTKCalculator(system; Ecut::Real, kgrid::Union{Nothing,<:AbstractVector{Int}}, tol=1e-6,
        temperature=zero(Real), verbose_scf=false, state=nothing
    )
    params = DFTKParameters(Ecut, kgrid, tol, temperature)

    if verbose_scf
        scf_callback=DFTK.ScfDefaultCallback()
    else
        scf_callback = (x) -> nothing
    end
    # Create dummy state if not given.
    if isnothing(state)
        state = DFTKState(system, params)
    end
    return DFTKCalculator(params, scf_callback, state)
end


AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(
        system::AbstractSystem, calculator::DFTKCalculator)
    # Create basis.
    basis = prepare_basis(system, calculator.params)
    scfres = self_consistent_field(basis, tol=calculator.params.tol, callback=calculator.scf_callback)
    calculator.state = DFTKState(scfres)
    return calculator.state.scfres.energies.total
end
AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(
        system::AbstractSystem, calculator::DFTKCalculator, state::DFTKState)
    # Update basis (and the enregy terms within).
    basis = prepare_basis(system, calculator.params)
    
    # Note that we use the state's densities and orbitals, but change the basis 
    # to reflect system changes.
    scfres = self_consistent_field(basis, ρ=state.scfres.ρ, ψ=state.scfres.ψ, tol=calculator.params.tol, callback=calculator.scf_callback)
    calculator.state = DFTKState(scfres)
    return calculator.state.scfres.energies.total
end
    
AtomsCalculators.@generate_interface function AtomsCalculators.forces(system::AbstractSystem, calculator::DFTKCalculator)
    basis = prepare_basis(system, calculator.params)
    scfres = self_consistent_field(basis, tol=calculator.params.tol, callback=calculator.scf_callback)
    calculator.state = DFTKState(scfres)
    return _compute_forces_cart(calculator.state.scfres)
end
AtomsCalculators.@generate_interface function AtomsCalculators.forces(system::AbstractSystem, calculator::DFTKCalculator, state::DFTKState)
    # Update basis (and the enregy terms within).
    basis = prepare_basis(system, calculator.params)
    
    # Note that we use the state's densities and orbitals, but change the basis 
    # to reflect system changes.
    scfres = self_consistent_field(basis, ρ=state.scfres.ρ, ψ=state.scfres.ψ, tol=calculator.params.tol, callback=calculator.scf_callback)
    calculator.state = DFTKState(scfres)
    return _compute_forces_cart(calculator.state.scfres)
end
