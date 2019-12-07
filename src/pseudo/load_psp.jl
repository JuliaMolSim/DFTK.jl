"""Return the data directory with pseudopotential files"""
datadir_psp() = joinpath(get(ENV, "DFTK_DATADIR", DFTK_DATADIR), "psp")


"""
    load_psp(identifier; datadir_psp)

Load a pseudopotential file from the library of pseudopotentials.
The file is searched in the directory `datadir_psp` and by the `identifier`.
If the identifier is a path to a valid file, the extension is used to determine
the type of the pseudopotential file format and a respective class is returned.
"""
function load_psp(identifier::AbstractString, datadir_psp=datadir_psp())
    identifier = lowercase(identifier)

    loadfctn = nothing
    extension = nothing
    if startswith(identifier, "hgh/") || endswith(identifier, ".hgh")
        parser = parse_hgh_file
        extension = "hgh"
    else
        error("Did not recognise pseudopotential type of path or identifier " *
              "$(identifier).")
    end

    # It is an existing file ... just parse it.
    isfile(identifier) && return parser(identifier)

    # It is not a file ... must be an identifier
    fullpath = joinpath(datadir_psp, identifier)

    # Add extension if not yet a file
    !isfile(fullpath) && (fullpath = fullpath * "." * extension)

    if !isfile(fullpath)
        error("Could not find pseudopotential matching identifier " *
              "'$identifier' in directory '$datadir_psp'")
    end

    return parser(fullpath, identifier=identifier)
end


"""
    list_psp(identifier; datadir_psp)

List all pseudopotential files known to DFTK. Allows to specify
part of an identifier to restrict the list

# Examples
```julia-repl
julia> list_psp("hgh")
```
will list all HGH-type pseudopotentials and
```julia-repl
julia> list_psp("hgh/lda")
```
will only list those for LDA (also known as Pade in this context).
"""
function list_psp(prefix::AbstractString="", datadir_psp=datadir_psp())
    prefix = lowercase(prefix)
    !endswith(prefix, "/") && prefix != "" && (prefix = prefix * "/")
    if !isdir(joinpath(datadir_psp, prefix))
        error("Prefix directory '$prefix' does not exist in '$datadir_psp'")
    end

    res = Vector{String}()
    for (root, dirs, files) in walkdir(joinpath(datadir_psp, prefix))
        root_relative = relpath(root, datadir_psp)
        append!(res, [joinpath(root_relative, f) for f in files])
    end

    # Ignore all fetch and update scripts
    [r for r in res if !endswith(r, ".sh")]
end
