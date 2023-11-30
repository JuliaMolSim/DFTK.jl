#
# Define AtomsCalculators interface for DFTK.
#
# This interface is inspired by the one used in Molly.jl, 
# see https://github.com/JuliaMolSim/Molly.jl/blob/master/src/types.jl


"""
    DFTKCalculator(; <keyword arguments>)

A calculator for use with the AtomsCalculators.jl interface.

`neighbors` can optionally be given as a keyword argument when calling the
calculation functions to save on computation when the neighbors are the same
for multiple calls.
In a similar way, `n_threads` can be given to determine the number of threads
to use when running the calculation function.

Not currently compatible with virial calculation.
Not currently compatible with using atom properties such as `σ` and `ϵ`.

# Arguments
- `Ecut::T`: kinetic energy cutoff for the `PlaneWaveBasis`.
- `kgrid::kgrid::Union{Nothing,Vec3{Int}}`: Number of k-points in each dimension.
    If not specified a grid is generated using `kgrid_from_minimal_spacing` with 
    a minimal spacing of `2π * 0.022` per Bohr.
- `tol`: Tolerance for the density change (``\|ρ_\text{out} - ρ_\text{in}\|``) in 
    the self-consistent field algorithm to flag convergence. Default is `1e-6`.
- `temperature::T`: If temperature==0, no fractional occupations are used. 
    If temperature is nonzero, the occupations are `fn = max_occ*smearing((εn-εF) / temperature)`.
"""
struct DFTKCalculator{T}
    Ecut::T
    kgrid::Union{Nothing,Vec3{Int}},
    tol
    temperature::T
end

function DFTKCalculator(;
        Ecut::T,
        kgrid::Union{Nothing,Vec3{Int}},
        tol=1e-6,
        temperature=zero(T),
    ) where {T <: Real}
    return DFTKCalculator(Ecut, kgrid, tol, temperature)
end

function warm_up_calculator(calculator::DFTKCalculator, system::AbstractSystem)
    model = model_LDA(system; temperature=calculator.temperature)
    basis = PlaneWaveBasis(model; calculator.Ecut, calculator.kgrid)
    return (model, basis)
end

AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(
        system::AbstractSystem, calculator::DFTKCalculator; precomputed=false, kwargs...)
    # If precomputed, use stored value, else compute and store.
    if precomputed
        return calculator.scfres.energies.total
    else
        _, basis = warm_up_calculator(calculator, system)
        calculator.scfres = self_consistent_field(basis, tol=calculator.tol)
        return calculator.scfres.energies.total
    end
end
    
AtomsCalculators.@generate_interface function AtomsCalculators.forces(
        system::AbstractSystem, calculator::DFTKCalculator; precomputed=false, cartesian=false, kwargs...)
    if cartesian
        _compute_forces = compute_forces
    else
        _compute_forces = compute_forces_cart
    end
    if precomputed
        return _compute_forces(calculator.scfres)
    else
        _, basis = warm_up_calculator(calculator, system)
        calculator.scfres = self_consistent_field(basis, tol=calculator.tol)
        return _compute_forces(calculator.scfres)
    end
end

# Re-implementation to avoid perfoming calculations twice. 
# This akward solution is enfocred by the interface of AtomsCalculators.
AtomsCalculators.@generate_interface function AtomsCalculators.energy_forces(
        system::AbstractSystem, calculator::DFTKCalculator; kwargs...)
    e = AtomsCalculators.potential_energy(system, calculator; kwargs...)
    f = AtomsCalculators.forces(system, calculator; kwargs..., precomputed=true) # Enforce precomputed = true
    return (;
        :energy => e,
        :forces => f
    )
end
