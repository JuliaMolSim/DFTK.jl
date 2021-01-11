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
    fs = split(filename,".")
    @assert length(fs) != 1 "No file extension mentioned"
    
    if(fs[end] == "vts")
        try
            save_scfres(filename, scfres, Val(:vtk))
        catch MethodError
            error("Package WriteVTK needs to be imported before using this function.")
        end
    elseif(fs[end] == "jld")
        try
            save_scfres(filename, scfres, Val(:jld))
        catch MethodError
            error("Package JLD2 needs to be imported before using this function.")
        end
    else
        error("$(fs[end]) extension is currently unsupported. The currently supported 
        extensions are .vts(for VTK file)  and .jld(for JLD file).")
    end
end

