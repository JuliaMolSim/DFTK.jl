"""Return the data directory with pseudopotential files"""
datadir_psp() = normpath(joinpath(@__DIR__, "..", "..", "data", "psp"))

"""
Load a pseudopotential file from the library of pseudopotentials.
The file is searched in the directory `datadir_psp()` and by the `key`.
If the `key` is a path to a valid file, the extension is used to determine
the type of the pseudopotential file format and a respective class is returned.
"""
function load_psp(key::AbstractString; kwargs...)
    if startswith(key, "hgh/") || endswith(lowercase(key), ".hgh")
        pseudo_type = PspHgh
        extension = ".hgh"
    elseif endswith(lowercase(key), ".upf")
        pseudo_type = PspUpf
        extension = ".upf"
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
        file::String = pseudofamily[element_symbol(atom)]
        get!(cached_psps, file) do
            load_psp(file; kwargs...)
        end
    end
end
