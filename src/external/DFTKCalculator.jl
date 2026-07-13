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
    #      state, but this may make it hard to keep track during geometry optimization
    #      or similar. In any case don't rely on this for now, it may disappear.
    counter_n_iter::Ref{Int}
    counter_matvec::Ref{Int}
    #
    # Calculator parameters
    enforce_convergence::Bool  # If true, throws an error exception on non-convergence
    derivatives_keep_model_symmetry::Bool # Enforces forces are always symmetric with respect
                               # to the structure, see docs of `compute_forces` for details

    function DFTKCalculator(params::DFTKParameters, st=nothing;
                            enforce_convergence=true, derivatives_keep_model_symmetry=false)
        new{Nothing}(params, st, Ref(0), Ref(0),
                     enforce_convergence, derivatives_keep_model_symmetry)
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
- `verbose::Bool` (default: `true`): If true, the SCF iterations are printed.
- `enforce_convergence::Bool` (default: `true`): If false, the calculator does not error out
  in case of a non-converging SCF.
- `derivatives_keep_model_symmetry::Bool` (default: `false`): If the parameters chosen for
  the discretization is not able to represent all symmetries of the structure one can either
  have (i) energy derivatives be consistent with the energy within the discretization used
  for the computation (ii) or have these derivatives agree with the physical model.
  See [`compute_forces`](@ref) for more details. By default we do (i), but setting this
  to true switches to (ii), which can be useful for geometry optimizations, for example.
  Using a `DFTKCalculator` in combination with
  [GeometryOptimization.jl](https://github.com/JuliaMolSim/GeometryOptimization.jl)
  automatically switches to (ii).

## Example
```julia-repl
julia> DFTKCalculator(; model_kwargs=(; functionals=LDA()),
                        basis_kwargs=(; Ecut=10, kgrid=(2, 2, 2)),
                        scf_kwargs=(; tol=1e-4))
```
without specifying a precise kgrid
```julia-repl
julia> DFTKCalculator(; model_kwargs=(; functionals=LDA()),
                        basis_kwargs=(; Ecut=10, kgrid=KgridSpacing(0.1)),
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
    DFTKCalculator(calc.params, st;
                   calc.enforce_convergence,
                   calc.derivatives_keep_model_symmetry)
end


function compute_scf(system::AbstractSystem, calc::DFTKCalculator, oldstate)
    # We re-use the symmetries from the oldstate to avoid issues if system
    # happens to be more symmetric than the structure used to make the oldstate.
    symmetries = haskey(oldstate, :basis) ? oldstate.basis.model.symmetries : true
    model = model_DFT(system; symmetries, calc.params.model_kwargs...)

    # Check if we can re-use the density / wavefunction from the state
    # or interpolate one to the other.
    ρ = nothing
    τ = nothing
    ψ = nothing
    basis = PlaneWaveBasis(model; calc.params.basis_kwargs...)
    if (haskey(oldstate, :basis) && haskey(oldstate, :ρ))
        lattice_agrees = maximum(abs, model.lattice - oldstate.basis.model.lattice) < 1e-6
        fft_size_agrees = (basis.fft_size..., model.n_spin_components) == size(oldstate.ρ)

        if lattice_agrees && fft_size_agrees
            @debug "compute_scf: Take ρ and ψ from oldstate"
            ρ = oldstate.ρ
            τ = oldstate.τ

            # Note: In principle the ψ may not be matching in size here ...
            ψ = get(oldstate, :ψ, nothing)
        else
            @debug "compute_scf: Interpolate ρ"
            ρ = interpolate_density(oldstate.ρ, oldstate.basis, basis)
            if any(needs_τ, basis.terms)
                τ = interpolate_density(oldstate.τ, oldstate.basis, basis)
            end
        end
    end
    if isnothing(ρ)
        @debug "compute_scf: Forming new guess density"
        ρ = guess_density(basis, system)
        if any(needs_τ, basis.terms)
            τ = zero(ρ)
        end
    end

    scfres = self_consistent_field(basis; ρ, τ, ψ, calc.params.scf_kwargs...)
    calc.enforce_convergence && !scfres.converged && error("SCF not converged.")
    calc.counter_n_iter[] += scfres.n_iter
    calc.counter_matvec[] += scfres.n_matvec
    scfres
end
function compute_scf(system::AbstractSystem, calc::DFTKCalculator, ::Nothing)
    compute_scf(system, calc, (; ))
end

function compute_calculator_forces(calc::DFTKCalculator, scfres)
    # By default forces are only symmetric with respect to the basis (discretized problem),
    # but not with respect to the model (original physical problem); see the compute_forces
    # docs for details; for geometry optimisations we need the latter, thus we may explicitly
    # symmetrise if the flag calc.derivatives_keep_model_symmetry is set.
    if calc.derivatives_keep_model_symmetry
        return _compute_forces_cart_symmetrized(scfres; scfres.basis.model.symmetries)
    else
        return compute_forces_cart(scfres)
    end
end

function compute_calculator_stresses(calc::DFTKCalculator, scfres)
    # By default stresses are only symmetric with respect to the basis (discretized problem),
    # but not with respect to the model (original physical problem); see the compute_stresses_cart
    # docs for details; for geometry optimisations we need the latter, thus we may explicitly
    # symmetrise if the flag calc.derivatives_keep_model_symmetry is set.
    if calc.derivatives_keep_model_symmetry
        return _compute_stresses_cart_symmetrized(scfres; scfres.basis.model.symmetries)
    else
        return compute_stresses_cart(scfres)
    end
end

@generate_interface function AtomsCalculators.calculate(::AtomsCalculators.Energy,
        system::AbstractSystem, calc::DFTKCalculator, ps=nothing, st=nothing; kwargs...)
    scfres = compute_scf(system, calc, st)
    (; energy=scfres.energies.total * u"hartree", state=scfres)
end

@generate_interface function AtomsCalculators.calculate(::AtomsCalculators.Forces,
        system::AbstractSystem, calc::DFTKCalculator, ps=nothing, st=nothing; kwargs...)
    scfres = compute_scf(system, calc, st)
    (; forces=compute_calculator_forces(calc, scfres) * u"hartree/bohr",
       energy=scfres.energies.total * u"hartree",
       state=scfres)
end

@generate_interface function AtomsCalculators.calculate(::AtomsCalculators.Virial,
        system::AbstractSystem, calc::DFTKCalculator, ps=nothing, st=nothing; kwargs...)
    scfres  = compute_scf(system, calc, st)
    Ω = scfres.basis.model.unit_cell_volume
    virial = (-Ω * compute_calculator_stresses(calc, scfres)) * u"hartree"
    (; virial, energy=scfres.energies.total * u"hartree", state=scfres)
end

function AtomsCalculators.energy_forces(system, calc::DFTKCalculator; kwargs...)
    AtomsCalculators.calculate(AtomsCalculators.Forces(), system, calc; kwargs...)
end

function AtomsCalculators.energy_forces_virial(system, calc::DFTKCalculator; kwargs...)
    res = AtomsCalculators.calculate(AtomsCalculators.Virial(), system, calc; kwargs...)
    (; forces=compute_calculator_forces(calc, res.state) * u"hartree/bohr", res...)
end
