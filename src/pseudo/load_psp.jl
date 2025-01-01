using PseudoPotentialData

"""Return the data directory with pseudopotential files"""
datadir_psp() = normpath(joinpath(@__DIR__, "..", "..", "data", "psp"))


extra_psp_kwargs(family::AbstractDict, element::Symbol) = (;)
function extra_psp_kwargs(family::PseudoFamily, element::Symbol)
    meta = pseudometa(family, element)
    haskey(meta, "rcut") ? (; rcut=meta["rcut"]) : (;)
end

"""
Load a pseudopotential file from a pseudopotential family.
This method should be preferred because it can automatically
use metadata from the pseudopotential family.
"""
function load_psp(family::AbstractDict, element::Symbol; kwargs...)
    load_psp(family[element]; extra_psp_kwargs(family, element)..., kwargs...)
end

"""
Load a pseudopotential file from the library of pseudopotentials.
The file is searched in the directory `datadir_psp()` and by the `key`.
If the `key` is a path to a valid file, the extension is used to determine
the type of the pseudopotential file format and a respective class is returned.
"""
function load_psp(key::AbstractString; kwargs...)
    if endswith(lowercase(key), ".gth")
        pseudo_type = PspHgh
        extension = ".gth"
    elseif endswith(lowercase(key), ".upf")
        pseudo_type = PspUpf
        extension = ".upf"
    elseif startswith(key, "hgh/") || endswith(lowercase(key), ".hgh")
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
        if startswith(identifier, datadir_psp())
            identifier = identifier[length(datadir_psp())+1:end]
        end
    else  # Not a file: treat as identifier, add extension if needed
        fullpath = joinpath(datadir_psp(), lowercase(key))
        isfile(fullpath) || (fullpath = fullpath * extension)
        identifier = replace(lowercase(key), "\\" => "/")
    end

    if isfile(fullpath)
        return pseudo_type(fullpath; identifier, kwargs...)
    else
        error("Could not find pseudopotential for identifier " *
              "'$identifier' in directory '$(datadir_psp())'")
    end
end

@deprecate(load_psp(dir::AbstractString, filename::AbstractString; kwargs...),
           load_psp(joinpath(dir, filename); kwargs...))

"""
Load all pseudopotentials from the pseudopotential family `pseudofamily`
corresponding to the atoms of a `system`. Returns the list of
the pseudopotential objects in the same order as the atoms in `system`.
Takes care that each pseudopotential object is only loaded once.
Applies the keyword arguments when loading all pseudopotentials.
`pseudofamily` can be a `PseudoPotentialData.PseudoFamily` or simply
a `Dict{Symbol,String}` which returns a file path when indexed
with an element symbol.
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
            load_psp(file; extra_psp_kwargs(pseudofamily, symbol)..., kwargs...)
        end
    end
end
