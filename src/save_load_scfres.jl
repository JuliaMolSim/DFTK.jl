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
    _, ext = splitext(filename)
    isempty(ext) && error("Provided filename misses extension")
    save_scfres(filename, scfres, Val(Symbol(ext[2:end])))
end

function save_scfres(filename::AbstractString, scfres::NamedTuple, format)
    error("The extension is not supported, which could indicate that a package"*
          "(e.g. JLD2 or WriteVTK) is not yet loaded.")
end
