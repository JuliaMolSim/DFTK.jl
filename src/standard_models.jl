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

If `functionals=[]` (empty list), then a reduced Hartree-Fock model is constructed.

All other keyword arguments
but `functional` are passed to [`model_atomic`](@ref) and from
there to [`Model`](@ref).

Note in particular that the `pseudopotential` keyword
argument is mandatory to specify pseudopotential information. This can be easily
achieved for example using the `PseudoFamily` struct from the `PseudoPotentialData`
package as shown below:

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
function _model_DFT(xc::Xc, args...; extra_terms=[],
                    coulomb_kernel_model::CoulombKernelModel=ProbeCharge(),
                    exx_strategy=CanonicalEXX(),
                    kwargs...)
    model_name = isempty(xc.functionals) ? "rHF" : join(string.(xc.functionals), "+")
    
    # handle exact exchange
    exx = [ExactExchange(; scaling_factor=exx_coefficient(f), coulomb_kernel_model, exx_strategy)
           for f in xc.functionals if !isnothing(exx_coefficient(f))]
    
    model_atomic(args...; extra_terms=[Hartree(), xc, exx..., extra_terms...], 
                 model_name, kwargs...)
end



"""
Build an Hartree-Fock model from the specified atoms.
"""
function model_HF(system::AbstractSystem; pseudopotentials, 
                  coulomb_kernel_model::CoulombKernelModel=ProbeCharge(), kwargs...)
    parsed = parse_system(system, pseudopotentials)
    _model_HF(parsed.lattice, parsed.atoms, parsed.positions; 
              coulomb_kernel_model, parsed.magnetic_moments, kwargs...)
end
function model_HF(lattice::AbstractMatrix, atoms::Vector{<:Element},
                  positions::Vector{<:AbstractVector}; 
                  coulomb_kernel_model::CoulombKernelModel=ProbeCharge(), kwargs...)
    _model_HF(lattice, atoms, positions; coulomb_kernel_model, kwargs...)
end
function _model_HF(args...; extra_terms=[], 
                   coulomb_kernel_model::CoulombKernelModel=ProbeCharge(), 
                   exx_strategy=CanonicalEXX(), 
                   kwargs...)
    model_atomic(args...; extra_terms=[Hartree(), ExactExchange(; coulomb_kernel_model, exx_strategy), extra_terms...],
                 model_name="HF", kwargs...)
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
Specify a PBE0 hybrid functional in conjunction with [`model_DFT`](@ref)
<https://doi.org/10.1063/1.478522>
"""
PBE0(; kwargs...) = Xc([:hyb_gga_xc_pbeh]; kwargs...)

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

"""
Specify a r2SCAN meta-GGA model in conjunction with [`model_DFT`](@ref)
<http://doi.org/10.1021/acs.jpclett.0c02405>
"""
r2SCAN(; kwargs...) = Xc([:mgga_x_r2scan, :mgga_c_r2scan]; kwargs...)


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
