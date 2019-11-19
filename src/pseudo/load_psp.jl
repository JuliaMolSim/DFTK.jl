function normalize_psp_identifier(identifier::AbstractString)
    normalized = lowercase(identifier)
    return replace(normalized, "lda" => "pade")
end


"""Return the data directory with pseudopotential files"""
datadir_psp() = joinpath(get(ENV, "DFTK_DATADIR", DFTK_DATADIR), "psp")


"""
    load_psp(identifier; datadir_psp)

Load a pseudopotential file from the library of pseudopotentials.
The file is searched in the directory `datadir_psp` and by the `identifier`.
"""
function load_psp(identifier::AbstractString, datadir_psp=datadir_psp())

    subdirs = []
    loadfctn = nothing
    kind = nothing
    if endswith(identifier, ".hgh")
        parser = parse_hgh_file
        push!(subdirs, "hgh")
        kind = "hgh"
    else
        error("Did not recognise pseudopotential type of path or identifier " *
              "$(identifier).")
    end

    if isfile(identifier)
        return parser(identifier)
    else
        identifier = normalize_psp_identifier(identifier)
        path = joinpath(datadir_psp, subdirs..., identifier)
        if !isfile(path)
            error("Could not find $(kind) pseudopotential matching identifier " *
                  "$(identifier). Searched at location $(path).")
        end
        return parser(path, identifier=identifier)
    end
end
