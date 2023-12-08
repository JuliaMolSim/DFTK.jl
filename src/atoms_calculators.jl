#
# Define AtomsCalculators interface for DFTK.
#
# This interface is inspired by the one used in Molly.jl, 
# see https://github.com/JuliaMolSim/Molly.jl/blob/master/src/types.jl
#
using AbstractFFTs: Plan
# TODO: Decide if calling potential_energy with a state should update the 
# state of the calculator (currently yes).

"""
    DFTKCalculator(; <keyword arguments>)

A calculator for use with the AtomsCalculators.jl interface.

# Arguments
- `Ecut::T`: kinetic energy cutoff for the `PlaneWaveBasis`.
- `kgrid::kgrid::Union{Nothing,Vec3{Int}}`: Number of k-points in each dimension.
    If not specified a grid is generated using `kgrid_from_minimal_spacing` with 
    a minimal spacing of `2π * 0.022` per Bohr.
- `tol`: Tolerance for the density change in 
    the self-consistent field algorithm to flag convergence. Default is `1e-6`.
- `temperature::T`: If temperature==0, no fractional occupations are used. 
    If temperature is nonzero, the occupations are `fn = max_occ*smearing((εn-εF) / temperature)`.
"""

using AtomsBase
using AtomsCalculators

struct DFTKState
    scfres::NamedTuple
end
function construct_DFTKState(basis)
    ρ = guess_density(basis)
    return DFTKState((;ρ, basis))
end

struct DFTKParameters
    Ecut::Real
    kgrid::Union{Nothing,<:AbstractVector{Int}}
    tol::Real
    temperature::Real
end

mutable struct DFTKCalculator <: AbstractCalculator
    params::DFTKParameters
    scf_callback
    state::DFTKState
end

function prepare_basis(system::AbstractSystem, params::DFTKParameters)
    model = model_LDA(system; temperature=params.temperature)
    basis = PlaneWaveBasis(model; params.Ecut, params.kgrid)
    return basis
end

function DFTKCalculator(system;
        Ecut::T,
        kgrid::Union{Nothing,<:AbstractVector{Int}},
        tol=1e-6,
        temperature=zero(T),
        verbose=false,
        state=nothing
    ) where {T <: Real}
    params = DFTKParameters(Ecut, kgrid, tol, temperature)

    if verbose
        scf_callback=DFTK.ScfDefaultCallback()
    else
        scf_callback = (x) -> nothing
    end
    # Create dummy state if not given.
    if isnothing(state)
        # By default create and LDA model.
        basis = prepare_basis(system, params)
        state = construct_DFTKState(basis)
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
    scfres = self_consistent_field(state.scfres.basis, ρ=state.scfres.ρ, tol=calculator.params.tol, callback=calculator.scf_callback)
    calculator.state = DFTKState(scfres)
    return calculator.state.scfres.energies.total
end
    
AtomsCalculators.@generate_interface function AtomsCalculators.forces(
        system::AbstractSystem, calculator::DFTKCalculator; cartesian=false)
    basis = prepare_basis(system, calculator.params)
    if cartesian
        _compute_forces = compute_forces
    else
        _compute_forces = compute_forces_cart
    end
    scfres = self_consistent_field(basis, tol=calculator.params.tol, callback=calculator.scf_callback)
    calculator.state = DFTKState(scfres)
    return _compute_forces(calculator.state.scfres)
end
AtomsCalculators.@generate_interface function AtomsCalculators.forces(
        system::AbstractSystem, calculator::DFTKCalculator, state::DFTKState; cartesian=false)
    scfres = self_consistent_field(state.scfres.basis, ρ=state.scfres.ρ, tol=calculator.params.tol, callback=calculator.scf_callback)
    calculator.state = DFTKState(scfres)
    if cartesian
        _compute_forces = compute_forces
    else
        _compute_forces = compute_forces_cart
    end
    return _compute_forces(calculator.state.scfres)
end
