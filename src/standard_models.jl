# High-level convenience functions to make standard models
# Note: When adding a function here, also add a method taking an AbstractSystem
#       to external/atomsbase.jl

"""
Convenience constructor, which builds a standard atomic (kinetic + atomic potential) model.
Use `extra_terms` to add additional terms.
"""
function model_atomic(lattice::AbstractMatrix,
                      atoms::Vector{<:Element},
                      positions::Vector{<:AbstractVector};
                      extra_terms=[], kinetic_blowup=BlowupIdentity(), kwargs...)
    @assert !(:terms in keys(kwargs))
    terms = [Kinetic(; blowup=kinetic_blowup),
             AtomicLocal(),
             AtomicNonlocal(),
             Ewald(),
             PspCorrection(),
             extra_terms...]
    if :temperature in keys(kwargs) && kwargs[:temperature] != 0
        terms = [terms..., Entropy()]
    end
    Model(lattice, atoms, positions; model_name="atomic", terms, kwargs...)
end


"""
Build a DFT model from the specified atoms, with the specified functionals.

# Examples
```julia-repl
julia> model_DFT(system, LDA(); temperature=0.01)
```
builds an [`LDA`](@ref) model for a passed system with specified smearing temperature.

```julia-repl
julia> model_DFT(system; functionals=LDA(), temperature=0.01)
```
Alternative syntax achieving exactly the same.
"""
function model_DFT(lattice::AbstractMatrix,
                   atoms::Vector{<:Element},
                   positions::Vector{<:AbstractVector},
                   xc::Xc;
                   extra_terms=[], kwargs...)
    model_name = isempty(xc.functionals) ? "rHF" : join(string.(xc.functionals), "+")
    model_atomic(lattice, atoms, positions;
                 extra_terms=[Hartree(), xc, extra_terms...], model_name, kwargs...)
end
function model_DFT(lattice::AbstractMatrix,
                   atoms::Vector{<:Element},
                   positions::Vector{<:AbstractVector},
                   functionals;
                   kwargs...)
    model_DFT(lattice, atoms, positions, Xc(functionals); kwargs...)
end
function model_DFT(lattice::AbstractMatrix,
                   atoms::Vector{<:Element},
                   positions::Vector{<:AbstractVector};
                   functionals, kwargs...)
    model_DFT(lattice, atoms, positions, functionals; kwargs...)
end


# Generate equivalent functions for AtomsBase
for fun in (:model_atomic, :model_DFT)
    @eval function $fun(system::AbstractSystem, args...; kwargs...)
        parsed = parse_system(system)
        $fun(parsed.lattice, parsed.atoms, parsed.positions, args...;
             parsed.magnetic_moments, kwargs...)
    end
end

#
# Convenient shorthands for frequently used functionals
#

"""
Specify an LDA model (Perdew & Wang parametrization) in conjunction with [`model_DFT`](@ref)
<https://doi.org/10.1103/PhysRevB.45.13244>
"""
LDA(; kwargs...) = Xc([:lda_x, :lda_c_pw]; kwargs...)

"""
Specify an PBE GGA model in conjunction with [`model_DFT`](@ref)
<https://doi.org/10.1103/PhysRevLett.77.3865>
"""
PBE(; kwargs...) = Xc([:gga_x_pbe, :gga_c_pbe]; kwargs...)

"""
Specify an PBEsol GGA model in conjunction with [`model_DFT`](@ref)
<https://doi.org/10.1103/physrevlett.100.136406>
"""
PBEsol(; kwargs...) = Xc([:gga_x_pbe_soL, :gga_c_pbe_sol]; kwargs...)

"""
Specify a SCAN meta-GGA model in conjunction with [`model_DFT`](@ref)
<https://doi.org/10.1103/PhysRevLett.115.036402>
"""
SCAN(; kwargs...) = Xc([:mgga_x_scan, :mgga_c_scan]; kwargs...)


@deprecate(model_LDA(lattice::AbstractMatrix, atoms::Vector{<:Element},
                     positions::Vector{<:AbstractVector}; kwargs...),
           model_DFT(lattice, atoms, positions, LDA(); kwargs...))
@deprecate(model_PBE(lattice::AbstractMatrix, atoms::Vector{<:Element},
                     positions::Vector{<:AbstractVector}; kwargs...),
           model_DFT(lattice, atoms, positions, PBE(); kwargs...))
@deprecate(model_SCAN(lattice::AbstractMatrix, atoms::Vector{<:Element},
                      positions::Vector{<:AbstractVector}; kwargs...),
           model_DFT(lattice, atoms, positions, SCAN(); kwargs...))
@deprecate model_LDA(system::AbstractSystem; kwargs...)  model_DFT(system, LDA();  kwargs...)
@deprecate model_PBE(system::AbstractSystem; kwargs...)  model_DFT(system, PBE();  kwargs...)
@deprecate model_SCAN(system::AbstractSystem; kwargs...) model_DFT(system, SCAN(); kwargs...)
