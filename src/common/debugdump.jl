using Preferences
using MPI

"""
Enable debug dumping; will dump a json file with some debug information for various
algorithms in the code; by default disabled. To enable call `set_debugdump_prefix!()`
or supply a custom location as a string argument. To disable call `set_debugdump_prefix!("")`.
"""
function set_debugdump_prefix!(prefix=joinpath(pwd(), "dftk-debug"))
    @set_preferences!("debugdump_prefix" => prefix)
    @info "debugdump_prefix preference changed to '$prefix'. Restart julia to see the effect."
end

"""Get current debug dumping prefix. A non-empty string means that debug dumping is enabled"""
debugdump_prefix() = @load_preference("debugdump_prefix", "")

"""
Save some dictionary data for a `dftkalgorithm` (e.g. `"fermialg"`). If `prefix` is an empty
string than debug dumping is disabled. Its default is controlled by the `"debugdump_prefix"`
local preference (to manage this preference, see the  `set_debugdump_prefix!` and
`debugdump_prefix` functions). By default hence, debug dumping is disabled.
"""
function save_debugdump(comm::MPI.Comm, dftkalgorithm::AbstractString, data::AbstractDict;
                        prefix=debugdump_prefix())
    if !isempty(prefix)
        save_debugdump(Val(:json), prefix, comm, dftkalgorithm, data)
    end
end
function save_debugdump(::Val, prefix::AbstractString, comm::MPI.Comm,
                        dftkalgorithm::AbstractString, data::AbstractDict)
    if mpi_master(comm)
        @warn("Saving debugdump requested, but not possible since dependent package not " *
              "loaded. Try 'using JSON3' in your script.")
    end
end
