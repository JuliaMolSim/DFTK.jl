import PeriodicTable
import PseudoPotentialIOExperimental: load_psp_file

"""Return the data directory with pseudopotential files"""
datadir_psp() = normpath(joinpath(@__DIR__, "..", "..", "data", "psp"))

"""
Load a pseudopotential file from the library of pseudopotentials.
The file is searched in the directory `datadir_psp()` and by the `key`.
If the `key` is a path to a valid file, the extension is used to determine
the type of the pseudopotential file format and a respective class is returned.
"""
function load_psp(key::AbstractString)
    if startswith(key, "hgh/") || endswith(lowercase(key), ".hgh")
        extension = ".hgh"
    else
        extension = splitext(key)[2]
    end
    if !isfile(key)
        fullpath = joinpath(datadir_psp(), lowercase(key))
        isfile(fullpath) || (fullpath = fullpath * extension)
    end
    isfile(fullpath) && return AtomicPotential(load_psp_file(fullpath))
    error("Could not resolve the filepath to $(key)")
end

function load_psp(dir::AbstractString, filename::AbstractString)
    isfile(joinpath(dir, filename)) && return load_psp(joinpath(dir, filename))
    return AtomicPotential(load_psp_file(dir, filename))
end
