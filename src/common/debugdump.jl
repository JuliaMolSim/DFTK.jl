using Preferences
using MPI

"""
Enable debug dumping; will dump a json file with some debug information for various
algorithms in the code; by default disabled.
"""
function set_debugdump_prefix!(prefix=joinpath(pwd(), "dftk-debug"))
    @set_preferences!("debugdump_prefix" => prefix)
    @info "debugdump_prefix preference changed. Restart julia to see the effect."
end

"""Check whether debug dumping is enabled"""
debugdump_enabled() = !isempty(@load_preference("debugdump_prefix", ""))

"""
Save some dictionary data for a `subsystem` (e.g. `"fermialg"`) in case
debug dumping is enabled
"""
function save_debugdump(comm::MPI.Comm, subsystem::AbstractString, data::AbstractDict)
    if debugdump_enabled()
        save_debugdump(Val(:json), comm, subsystem, data)
    end
end
function save_debugdump(::Val, comm::MPI.Comm, subsystem::AbstractString, data::AbstractDict)
    if mpi_master(comm)
        @warn("Saving debugdump not possible since dependent package not loaded. " *
              "Try 'using JSON3' in your script.")
    end
end
