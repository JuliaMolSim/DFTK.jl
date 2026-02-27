# High-level convenience functions to make standard models
# Note: When adding a function here, also add a method taking an AbstractSystem
#       to external/atomsbase.jl

"""
Convenience constructor around [`Model`](@ref),
which builds a standard atomic (kinetic + atomic potential) model.

## Keyword arguments
- `pseudopotentials`: Set the pseudopotential information for the atoms
   of the passed system. Can be (a) a list of pseudopotential objects
   (one for each atom), where a `nothing` element indicates that the
   Coulomb potential should be used for that atom or (b)
   a `PseudoPotentialData.PseudoFamily` to automatically determine the
   pseudopotential from the specified pseudo family or (c)
   a `Dict{Symbol,String}` mapping an atomic symbol
   to the pseudopotential to be employed.
- `extra_terms`: Specify additional terms to be passed to the
  [`Model`](@ref) constructor.
- `kinetic_blowup`: Specify a blowup function for the kinetic
  energy term, see e.g [`BlowupCHV`](@ref).

# Examples
```julia-repl
julia> model_atomic(system; pseudopotentials=PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf"))
```
Construct an atomic system using the specified pseudo-dojo pseudopotentials for all
atoms of the system.

```julia-repl
julia> model_atomic(system; pseudopotentials=Dict(:Si => "path/to/pseudofile.upf"))
```
same thing, but specify the pseudopotential path explicitly in a dictionary.
"""
function model_atomic(system::AbstractSystem; pseudopotentials, kwargs...)
    # Note: We are enforcing to specify pseudopotentials at this interface
    # (unlike the lower-level Model interface) because the argument is that
    # automatically defaulting to the Coulomb potential will generally trip
    # people over and could too easily lead to garbage results
    #
    parsed = parse_system(system, pseudopotentials)
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
For the most important standard functionals, convenience wrappers can be
used to directly pass the right set of arguments to the `functionals` keyword.
See for example [`LDA`](@ref), [`PBE`](@ref), [`PBEsol`](@ref), [`SCAN`](@ref),
[`r2SCAN`](@ref), [`PBE0`](@ref).

If `functionals=[]` (empty list), then a reduced Hartree-Fock model is constructed.

All other keyword arguments
but `functional` are passed to [`model_atomic`](@ref) and from
there to [`Model`](@ref).

Note in particular that the `pseudopotential` keyword
argument is mandatory to specify pseudopotential information. This can be easily
achieved for example using the `PseudoFamily` struct from the `PseudoPotentialData`
package as shown below:

!!! warn "Hybrid DFT is experimental"
         The interface for Hybrid DFT models may change at any moment,
         which is not considered a breaking change.
         Note further that at this stage (Feb 2026) there are still
         known performance bottle necks in the code.

# Examples
```julia-repl
julia> model_DFT(system; functionals=LDA(), temperature=0.01,
                 pseudopotentials=PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf"))

```
builds an [`LDA`](@ref) model for a passed system
with specified smearing temperature.

```julia-repl
julia> model_DFT(system; functionals=[:lda_x, :lda_c_pw], temperature=0.01,
                 pseudopotentials=PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf"))
```
Alternative syntax specifying the functionals directly
via their libxc codes.
"""
function model_DFT(system::AbstractSystem; pseudopotentials, functionals, kwargs...)
    # Note: We are deliberately enforcing the user to specify pseudopotentials here.
    # See the implementation of model_atomic for a rationale why
    #
    # TODO Could check consistency between pseudos and passed functionals
    parsed = parse_system(system, pseudopotentials)
    _model_DFT(functionals, parsed.lattice, parsed.atoms, parsed.positions;
               parsed.magnetic_moments, kwargs...)
end
function model_DFT(lattice::AbstractMatrix, atoms::Vector{<:Element},
                   positions::Vector{<:AbstractVector}; functionals, kwargs...)
    _model_DFT(functionals, lattice, atoms, positions; kwargs...)
end
function _model_DFT(functionals::AbstractVector, args...; extra_terms=[], kwargs...)
    (; model_name, dftterms) = _parse_functionals(functionals)
    model_atomic(args...;
                 extra_terms=[Hartree(), dftterms..., extra_terms...], model_name, kwargs...)
end
_model_DFT(xc::Xc, args...; kwargs...) = _model_DFT([xc], args...; kwargs...)

function _parse_functionals(functionals::AbstractVector)
    # The idea is for the functionals keyword argument to be pretty smart in the long run,
    # such that things like
    #  - `model_DFT(system; functionals=B3LYP())`
    #  - `model_DFT(system; functionals=[LibxcFunctional(:lda_x)])`
    #  - `model_DFT(system; functionals=[:lda_x, :lda_c_pw, HubbardU(data)])`
    #  - `model_DFT(system; functionals=[:hyb_gga_xc_pbeh])'
    # will all work.
    # This function does the parsing and returns the terms and model_name to be used
    # with model_atomic

    exx  = nothing
    xc   = nothing
    rest = eltype(functionals)[]
    for fun in functionals
        if fun isa ExactExchange
            exx = fun
        elseif fun isa Xc
            xc = fun
        else
            push!(rest, fun)
        end
    end

    if !isnothing(xc) && !isempty(rest)
        throw(ArgumentError("Cannot provide both xc object and constituent functionals " *
        "to functionals keyword."))
    end
    xc = @something xc Xc(rest)
    if isempty(xc.functionals) && !isnothing(exx)
        throw(ArgumentError("Cannot use model_DFT to construct Hartree-Fock models. " * 
                            "Use model_HF for this purpose."))
    end

    # Add hybrid functional if functional is hybrid, but no EXX given so far.
    exx_coeffs = filter(!isnothing, map(exx_coefficient, xc.functionals))
    if isnothing(exx) && !isempty(exx_coeffs)
        exx = ExactExchange(; scaling_factor=only(exx_coeffs))
    end

    if isempty(xc.functionals)
        @assert isnothing(exx)
        model_name = "rHF"
    else
        model_name = join(string.(xc.functionals), "+")
    end
    (; model_name, dftterms=filter(!isnothing, [xc, exx]))
end


"""
Build an Hartree-Fock model from the specified atoms.

!!! warn "Hartree-Fock is experimental"
         The interface may change at any moment, which is not considered a breaking change.
         Note further that at this stage (Feb 2026) there are still known performance bottle
         necks in the code.
"""
function model_HF(system::AbstractSystem; pseudopotentials,
                  interaction_kernel::InteractionKernel=Coulomb(),
                  exx_algorithm::ExxAlgorithm=VanillaExx(), extra_terms=[], kwargs...)
    # Note: We are deliberately enforcing the user to specify pseudopotentials here.
    # See the implementation of model_atomic for a rationale why
    #
    exx = ExactExchange(; interaction_kernel, exx_algorithm)
    model_atomic(system; pseudopotentials, model_name="HF",
                 extra_terms=[Hartree(), exx, extra_terms...], kwargs...)
end
function model_HF(lattice::AbstractMatrix, atoms::Vector{<:Element},
                  positions::Vector{<:AbstractVector};
                  interaction_kernel::InteractionKernel=Coulomb(),
                  exx_algorithm::ExxAlgorithm=VanillaExx(), extra_terms=[], kwargs...)
    exx = ExactExchange(; interaction_kernel, exx_algorithm)
    model_atomic(lattice, atoms, positions; model_name="HF",
                 extra_terms=[Hartree(), exx, extra_terms...], kwargs...)
end


#
# Convenient shorthands for frequently used functionals
#

"""
Specify an LDA model (Perdew & Wang parametrization) in conjunction with [`model_DFT`](@ref)
<https://doi.org/10.1103/PhysRevB.45.13244>.
Possible keyword arguments are those accepted by [`Xc`](@ref).
"""
LDA(; kwargs...) = Xc([:lda_x, :lda_c_pw]; kwargs...)

"""
Specify an PBE GGA model in conjunction with [`model_DFT`](@ref)
<https://doi.org/10.1103/PhysRevLett.77.3865>.
Possible keyword arguments are those accepted by [`Xc`](@ref).
"""
PBE(; kwargs...) = Xc([:gga_x_pbe, :gga_c_pbe]; kwargs...)

"""
Specify an PBEsol GGA model in conjunction with [`model_DFT`](@ref)
<https://doi.org/10.1103/physrevlett.100.136406>.
Possible keyword arguments are those accepted by [`Xc`](@ref).
"""
PBEsol(; kwargs...) = Xc([:gga_x_pbe_sol, :gga_c_pbe_sol]; kwargs...)

"""
Specify a SCAN meta-GGA model in conjunction with [`model_DFT`](@ref)
<https://doi.org/10.1103/PhysRevLett.115.036402>.
Possible keyword arguments are those accepted by [`Xc`](@ref).
"""
SCAN(; kwargs...) = Xc([:mgga_x_scan, :mgga_c_scan]; kwargs...)

"""
Specify a r2SCAN meta-GGA model in conjunction with [`model_DFT`](@ref)
<http://doi.org/10.1021/acs.jpclett.0c02405>.
Possible keyword arguments are those accepted by [`Xc`](@ref).
"""
r2SCAN(; kwargs...) = Xc([:mgga_x_r2scan, :mgga_c_r2scan]; kwargs...)

"""
Specify a PBE0 hybrid functional in conjunction with [`model_DFT`](@ref)
<https://doi.org/10.1063/1.478522>
Possible keyword arguments are those accepted by [`Xc`](@ref) and by
[`ExactExchange`](@ref). Use the keyword argument `exx_fraction` to specify a
custom exact exchange fraction.

!!! warn "Hybrid DFT is experimental"
         The interface may change at any moment, which is not considered a breaking change.
         Note further that at this stage (Feb 2026) there are still known performance bottle
         necks in the code.
"""
PBE0(; kwargs...)  = HybridFunctional([:hyb_gga_xc_pbeh]; kwargs...)


"""
Specify a HSE hybrid functional in conjunction with [`model_DFT`](@ref)
<https://doi.org/10.1063/1.2404663>
Possible keyword arguments are those accepted by [`Xc`](@ref) and by
[`ExactExchange`](@ref). Use the keyword argument `exx_fraction` to specify a
custom exact exchange fraction.

This is the HSE06 hybrid functional with range-separation parameter μ=0.11/bohr.

Note that other codes use slightly different μ:
* VASP uses μ = 0.2/Angstrom = 0.105835/bohr
* Quantum Espresso uses μ=0.106/bohr if input_dft='hse'
* Quantum Espresso uses μ=0.11/bohr  if input_dft='xc-000i-000i-000i-428l'

!!! warn "Hybrid DFT is experimental"
         The interface may change at any moment, which is not considered a breaking change.
         Note further that at this stage (Feb 2026) there are still known performance bottle
         necks in the code.
"""
HSE(; kwargs...) = HybridFunctional([:hyb_gga_xc_hse06]; 
                                    exx_fraction=0.25,  # have to pass as range-separated hybrids don't provide exx fraction
                                    interaction_kernel=ErfShortRangeCoulomb(μ=0.11),  
                                    kwargs...)

# Internal function to help define hybrid functional shorthands
function HybridFunctional(libxc_symbols;
                          exx_fraction=nothing,
                          interaction_kernel::InteractionKernel=Coulomb(),
                          exx_algorithm::ExxAlgorithm=VanillaExx(), kwargs...)
    xc  = Xc(libxc_symbols; kwargs...)
    scaling_factor = @something(exx_fraction, begin
        only(filter(!isnothing, map(exx_coefficient, xc.functionals)))
    end)

    exx = ExactExchange(; scaling_factor, interaction_kernel, exx_algorithm)
    [xc, exx]
end

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
