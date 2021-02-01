function gather_kpts_scfres(scfres::NamedTuple)
    # TODO Rename this to gather_kpts once scfres gets its proper type

    # Need gathering over k-Points:
    kpt_properties = (:ψ, :occupation, :eigenvalues)
    scfdict = Dict{Symbol, Any}()
    for symbol in kpt_properties
        scfdict[symbol] = gather_kpts(getproperty(scfres, symbol), scfres.basis)
    end
    scfdict[:basis] = gather_kpts(scfres.basis)

    if mpi_master()
        (; (symbol => get(scfdict, symbol, getproperty(scfres, symbol))
            for symbol in propertynames(scfres))...)
    else
        nothing
    end
end


"""
    load_scfres(filename)

Load back an `scfres`, which has previously been stored with `save_scfres`.
Note the warning in `save_scfres`.
"""
function load_scfres end


"""
    save_scfres(filename, scfres)

Save an `scfres` obtained from `self_consistent_field` to a JLD2 
or VTK file depending on the extension.

!!! warning "No compatibility guarantees"
    No guarantees are made with respect to this function at this point.
    It may change incompatibly between DFTK versions or stop working / be removed
    in the future.
"""
function save_scfres(filename::AbstractString, scfres::NamedTuple; kwargs...)
    _, ext = splitext(filename)
    ext = Symbol(ext[2:end])

    # Whitelist valid extensions
    !(ext in (:jld2, :vts)) && error("Extension '$ext' not supported by DFTK.")

    # Gather scfres data on master MPI process
    scfres = gather_kpts_scfres(scfres)

    # On master MPI process dispatch to individual functions based on extension
    ret = isnothing(scfres) ? nothing : save_scfres_master(filename, scfres, Val(ext); kwargs...)

    # Return from this function synchronously on all processes
    # (to ensure that data has actually been written upon return)
    MPI.Barrier(MPI.COMM_WORLD)
    ret
end


function save_scfres_master(filename::AbstractString, ::NamedTuple, ::Any; kwargs...)
    error("The extension $(last(splitext(filename))) is currently not available. " *
          "A required package (e.g. JLD2 or WriteVTK) is not yet loaded.")
end
