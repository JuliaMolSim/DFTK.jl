"""Return the data directory with pseudopotential files"""
datadir_psp() = normpath(joinpath(@__DIR__, "..", "..", "data", "psp"))

extra_pseudometa_kwargs(::AbstractDict, ::Symbol) = NamedTuple()
function extra_pseudometa_kwargs(family::PseudoFamily, element::Symbol)
    meta = pseudometa(family, element)
    haskey(meta, "rcut") ? (; rcut=meta["rcut"]) : NamedTuple()
end

"""
Load a pseudopotential file from a pseudopotential family.
Uses available metadata from the pseudopotential family
(via the `pseudometa` function of `PseudoPotentialData`)
to automatically set some keyword arguments.
`pseudofamily` can be a `PseudoPotentialData.PseudoFamily` or simply
a `Dict{Symbol,String}` which returns a file path when indexed
with an element symbol.

## Example
```julia
using PseudoPotentialData
pseudopotentials = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
load_psp(pseudopotenitals, :Si)
```
"""
function load_psp(family::AbstractDict, element::Symbol; kwargs...)
    load_psp(family[element]; extra_pseudometa_kwargs(family, element)..., kwargs...)
end

"""
Load all pseudopotentials from the pseudopotential family `pseudofamily`
corresponding to the atoms of a `system`. Returns the list of
the pseudopotential objects in the same order as the atoms in `system`.
Takes care that each pseudopotential object is only loaded once
(which enables later efficiency improvements in DFTK).
Applies the passed keyword arguments when loading all pseudopotentials
and additionally uses the metadata stored for each pseudopotential family
to deduce further keyword arguments (e.g. `rcut`).
`pseudofamily` can be a `PseudoPotentialData.PseudoFamily` or simply
a `Dict{Symbol,String}` which returns a file path when indexed
with an element symbol.

## Example
```julia
using PseudoPotentialData
using AtomsBuilder
pseudopotentials = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")
load_psp(pseudopotenitals, bulk(:Si))
```
"""
function load_psp(pseudofamily::AbstractDict{Symbol,<:AbstractString},
                  system::AbstractSystem; kwargs...)
    # Cache for instantiated pseudopotentials. This is done to ensure that identical
    # pseudos are indistinguishable in memory, which is used in the Model constructor
    # to deduce the atom_groups.
    cached_psps = Dict{String, Any}()
    map(system) do atom
        symbol = element_symbol(atom)
        file::String = pseudofamily[symbol]
        get!(cached_psps, file) do
            load_psp(file; extra_pseudometa_kwargs(pseudofamily, symbol)..., kwargs...)
        end
    end
end

"""
Load a pseudopotential file. The file extension is used to determine
the type of the pseudopotential file format and a respective class
(e.g. `PspHgh` or `PspUpf`) is returned. Most users will want to use
other methods of the `load_psp` function.
"""
function load_psp(key::AbstractString; kwargs...)
    if endswith(lowercase(key), ".gth")
        pseudo_type = PspHgh
        extension = ".gth"
    elseif endswith(lowercase(key), ".upf")
        pseudo_type = PspUpf
        extension = ".upf"
    elseif startswith(lowercase(key), "hgh/") || endswith(lowercase(key), ".hgh")
        # TODO Legacy block still needed for GTH pseudos bundled with DFTK
        pseudo_type = PspHgh
        extension = ".hgh"
    else
        error("Could not determine pseudopotential family of '$key'")
    end

    Sys.iswindows() && (key = replace(key, "/" => "\\"))
    if isfile(key)  # Key is a file ... deduce identifier
        fullpath = key
        identifier = replace(key, "\\" => "/")
    else  # Not a file: treat as identifier, add extension if needed
        @warn("Calling `load_psp` without specifying a full path to a pseudopotential file " *
              "(i.e. identifiers such as hgh/lda/Si-q4) are deprecated as DFTK's internal " *
              "pseudopotential library will be removed in the future. Please use the " *
              "PseudoPotentialData package to supply pseudopotentials to DFTK. (e.g. here " *
              "`load_psp(PseudoFamily(\"cp2k.nc.sr.lda.v0_1.semicore.gth\"), :Si)`)")
        fullpath = joinpath(datadir_psp(), lowercase(key))
        isfile(fullpath) || (fullpath = fullpath * extension)
        identifier = replace(lowercase(key), "\\" => "/")
    end

    if isfile(fullpath)
        return pseudo_type(fullpath; identifier, kwargs...)
    else
        error("Could not find pseudopotential file '$key'")
    end
end

@deprecate(load_psp(dir::AbstractString, filename::AbstractString; kwargs...),
           load_psp(joinpath(dir, filename); kwargs...))
