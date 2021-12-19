import PeriodicTable

"""Return the data directory with pseudopotential files"""
datadir_psp() = joinpath(get(ENV, "DFTK_DATADIR", DFTK_DATADIR), "psp")


"""
Load a pseudopotential file from the library of pseudopotentials.
The file is searched in the directory `datadir_psp()` and by the `key`.
If the `key` is a path to a valid file, the extension is used to determine
the type of the pseudopotential file format and a respective class is returned.
"""
function load_psp(key::AbstractString; identifier=nothing)
    if startswith(key, "hgh/") || endswith(key, ".hgh")
        parser = parse_hgh_file
        extension = ".hgh"
    else
        error("Could not determine pseudopotential family of '$key'")
    end

    Sys.iswindows() && (key = replace(key, "/" => "\\"))
    if isfile(key) # Key is a file ... deduce identifier
        fullpath = key
        if isnothing(identifier)
            identifier = replace(key, "\\" => "/")
            if startswith(identifier, datadir_psp())
                identifier = identifier[length(datadir_psp())+1:end]
            end
        end
    else
        # Not a file: Treat as identifier, add extension if needed
        @assert isnothing(identifier)
        fullpath = joinpath(datadir_psp(), lowercase(key))
        isfile(fullpath) || (fullpath = fullpath * extension)
        identifier = replace(lowercase(key), "\\" => "/")
    end

    if isfile(fullpath)
        return parser(fullpath, identifier=identifier)
    else
        error("Could not find pseudopotential for identifier " *
              "'$identifier' in directory '$(datadir_psp())'")
    end
end


function load_psp(element::Union{Symbol,Integer}; family="hgh", core=:fullcore, kwargs...)
    list = list_psp(element; family=family, core=core, kwargs...)
    if length(list) != 1
        error("Parameters passed to load_psp do not uniquely identify a PSP file.")
    end
    load_psp(list[1].path, identifier=list[1].identifier)
end
