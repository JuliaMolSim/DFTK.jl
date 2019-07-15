function normalize_psp_identifier(identifier::AbstractString)
    normalized = lowercase(identifier)
    return replace(normalized, "lda" => "pade")
end


"""
    load_psp(identifier; search_directory)

Load a pseudopotential file from the library of pseudopotentials.
The file is searched in the directory `search_directory` and by the `identifier`.
"""
function load_psp(identifier::AbstractString,
                  search_directory=joinpath(get(ENV, "DFTK_DATADIR", DFTK_DATADIR), "psp"))

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
        path = joinpath(search_directory, subdirs..., identifier)
        if !isfile(path)
            error("Could not find $(kind) pseudopotential matching identifier " *
                  "$(identifier). Searched at location $(path).")
        end
        return parser(path, identifier=identifier)
    end
end
