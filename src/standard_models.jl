# High-level convenience functions to make standard models
# Note: When adding a function here, also add a method taking an AbstractSystem
#       to external/atomsbase.jl

"""
Convenience constructor around [`Model`](@ref),
which builds a standard atomic (kinetic + atomic potential) model.

## Keyword arguments
- `extra_terms`: Specify additional terms to be passed to the
  [`Model`](@ref) constructor.
- `kinetic_blowup`: Specify a blowup function for the kinetic
  energy term, see e.g [`BlowupCHV`](@ref).

"""
function model_atomic(system::AbstractSystem; kwargs...)
    parsed = parse_system(system)
    model_atomic(parsed.lattice, parsed.atoms, parsed.positions;
                 parsed.magnetic_moments, kwargs...)
end
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
Build a DFT model from the specified atoms with the specified XC functionals.

The `functionals` keyword argument takes either an [`Xc`](@ref) object,
a list of objects subtyping `DftFunctionals.Functional` or a list of
`Symbol`s. For the latter any [functional symbol from libxc](https://libxc.gitlab.io/functionals/)
can be specified, see examples below.
Note, that most DFT models require two symbols in the `functionals` list
(one for the exchange and one for the correlation part).

If `functionals=[]` (empty list), then a reduced Hartree-Fock model is constructed.

All other keyword arguments
but `functional` are passed to [`model_atomic`](@ref) and from
there to [`Model`](@ref).

# Examples
```julia-repl
julia> model_DFT(system; functionals=LDA(), temperature=0.01)
```
builds an [`LDA`](@ref) model for a passed system
with specified smearing temperature.

```julia-repl
julia> model_DFT(system; functionals=[:lda_x, :lda_c_pw], temperature=0.01)
```
Alternative syntax specifying the functionals directly
via their libxc codes.
"""
function model_DFT(system::AbstractSystem; functionals, kwargs...)
    parsed = parse_system(system)
    model_DFT(parsed.lattice, parsed.atoms, parsed.positions;
              parsed.magnetic_moments, functionals, kwargs...)
end
function model_DFT(lattice::AbstractMatrix,
                   atoms::Vector{<:Element},
                   positions::Vector{<:AbstractVector};
                   functionals, kwargs...)
    # The idea is for the functionals keyword argument to be pretty smart in the long run,
    # such that things like
    #  - `model_DFT(system; functionals=B3LYP())`
    #  - `model_DFT(system; functionals=[LibxcFunctional(:lda_x)])`
    #  - `model_DFT(system; functionals=[:lda_x, :lda_c_pw, HubbardU(data)])`
    # will all work.
    _model_DFT(functionals, lattice, atoms, positions; kwargs...)
end
function _model_DFT(functionals::AbstractVector, args...; kwargs...)
    _model_DFT(Xc(functionals), args...; kwargs...)
end
function _model_DFT(xc::Xc, args...; extra_terms=[], kwargs...)
    model_name = isempty(xc.functionals) ? "rHF" : join(string.(xc.functionals), "+")
    model_atomic(args...; extra_terms=[Hartree(), xc, extra_terms...], model_name, kwargs...)
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
PBEsol(; kwargs...) = Xc([:gga_x_pbe_sol, :gga_c_pbe_sol]; kwargs...)

"""
Specify a SCAN meta-GGA model in conjunction with [`model_DFT`](@ref)
<https://doi.org/10.1103/PhysRevLett.115.036402>
"""
SCAN(; kwargs...) = Xc([:mgga_x_scan, :mgga_c_scan]; kwargs...)


@deprecate(model_LDA(lattice::AbstractMatrix, atoms::Vector{<:Element},
                     positions::Vector{<:AbstractVector}; kwargs...),
           model_DFT(lattice, atoms, positions; functionals=LDA(), kwargs...))
@deprecate(model_PBE(lattice::AbstractMatrix, atoms::Vector{<:Element},
                     positions::Vector{<:AbstractVector}; kwargs...),
           model_DFT(lattice, atoms, positions; functionals=PBE(), kwargs...))
@deprecate(model_SCAN(lattice::AbstractMatrix, atoms::Vector{<:Element},
                      positions::Vector{<:AbstractVector}; kwargs...),
           model_DFT(lattice, atoms, positions; functionals=SCAN(), kwargs...))
@deprecate(model_DFT(lattice::AbstractMatrix, atoms::Vector{<:Element},
                   positions::Vector{<:AbstractVector}, functionals; kwargs...),
           model_DFT(lattice, atoms, positions; functionals, kwargs...))

@deprecate model_LDA(system::AbstractSystem; kwargs...)  model_DFT(system; functionals=LDA(),  kwargs...)
@deprecate model_PBE(system::AbstractSystem; kwargs...)  model_DFT(system; functionals=PBE(),  kwargs...)
@deprecate model_SCAN(system::AbstractSystem; kwargs...) model_DFT(system; functionals=SCAN(), kwargs...)
@deprecate(model_DFT(system::AbstractSystem, functionals; kwargs...),
           model_DFT(system; functionals, kwargs...))
