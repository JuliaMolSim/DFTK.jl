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
the type of the pseudopotential file format and a corresponding class is returned.
The following formats are supported:
- Goedecker-Teter-Hutter (.gth / .hgh files), yielding a `PspHgh`;
- Unified Pseudopotential Format (.upf files), yielding a `PspUpf`;
- Abinit pseudopotential format 8 (.psp8 files), yielding a `PspUpf`.

Most users will want to use other methods of the `load_psp` function.
"""
function load_psp(fullpath::AbstractString; kwargs...)
    if !isfile(fullpath) 
        error("Could not find pseudopotential file '$fullpath'. Note: The DFTK-bundled" *
              "pseudopotentials (keys 'hgh/pbe/c-q4' and similar) have been removed in" *
              "DFTK 0.8.1. Please use PseudoPotentialData to select pseudopotentials." *
              "See the DFTK tutorial and documentation for details.")
    end

    # TODO: We keep this identifier in the form it was introduced during a time
    #       DFTK had still a built-in pseudopotential library.
    identifier = Sys.iswindows() ? replace(fullpath, "/" => "\\") : fullpath

    fullpath_lc = lowercase(fullpath)
    if endswith(fullpath_lc, ".gth")
        return PspHgh(fullpath; identifier, kwargs...)
    elseif endswith(fullpath_lc, ".upf") || endswith(fullpath_lc, ".psp8")
        # PspUpf has a constructor that will convert from a Psp8File to a UpfFile
        return PspUpf(fullpath; identifier, kwargs...)
    else
        error("Could not determine pseudopotential family of '$fullpath'")
    end
end
