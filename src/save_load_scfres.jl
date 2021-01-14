"""
    load_scfres(filename)

Load back an `scfres`, which has previously been stored with `save_scfres`.
Note the warning in `save_scfres`.
"""
function load_scfres end



"""
    save_scfres(filename, scfres)

Save an `scfres` obtained from `self_consistent_field` to a JLD2 or VTK file depending on the extension.

!!! warning "No compatibility guarantees"
    No guarantees are made with respect to this function at this point.
    It may change incompatibly between DFTK versions or stop working / be removed
    in the future.
"""
function save_scfres(filename::AbstractString, scfres::NamedTuple)
    fs = splitext(filename)
    @assert length(fs) != 1 "No file extension mentioned"
    save_scfres(filename, scfres, Val(Symbol(fs[end])))
end

function save_scfres(filename::AbstractString, scfres::NamedTuple, format)
    if format == Val(Symbol(".vts"))
        error("Package WriteVTK needs to be imported before using this function.")
    elseif format == Val(Symbol(".jld"))
        error("Package JLD2 needs to be imported before using this function.")
    else
        error("$(splitext(filename)[end])extension is currently unsupported. The currently supported 
        extensions are .vts(for VTK file)  and .jld(for JLD file).")
    end
end
