import PeriodicTable

"""Return the data directory with pseudopotential files"""
datadir_psp() = joinpath(get(ENV, "DFTK_DATADIR", DFTK_DATADIR), "psp")


"""
    load_psp(identifier; datadir_psp)

Load a pseudopotential file from the library of pseudopotentials.
The file is searched in the directory `datadir_psp` and by the `identifier`.
If the identifier is a path to a valid file, the extension is used to determine
the type of the pseudopotential file format and a respective class is returned.
"""
function load_psp(key::AbstractString; datadir_psp=datadir_psp())
    if startswith(key, "hgh/") || endswith(key, ".hgh")
        parser = parse_hgh_file
        extension = ".hgh"
    else
        error("Could not determine pseudopotential family of '$key'")
    end
    isfile(key) && return parser(key)

    # Not a file: Threat as identifier, add extension if needed
    identifier = lowercase(key)
    fullpath = joinpath(datadir_psp, identifier)
    isfile(fullpath) || (fullpath = fullpath * extension)

    if isfile(fullpath)
        parser(fullpath, identifier=identifier)
    else
        error("Could not find pseudopotential for identifier " *
              "'$identifier' in directory '$datadir_psp'")
    end
end


function load_psp(element::Union{Symbol,Integer}; family="hgh", core=:fullcore, kwargs...)
    list = list_psp(element; family=family, core=core, kwargs...)
    if length(list) != 1
        error("Parameters passed to load_psp do not uniquely identify a PSP file.")
    end
    load_psp(list[1].path)
end
