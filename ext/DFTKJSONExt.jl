module DFTKJSONExt
using DFTK
using JSON
using MPI
using Preferences

# JSON.jl serialises multidimensional arrays as nested arrays, while the previously used
# JSON3 package stored them as flat (column-major) arrays. This series of functions
# explicitly flattens arrays to make sure the resulting json files when using JSON and JSON3
# are identical to avoid downstream issues for now.
flatten_arrays(x) = x
function flatten_arrays(x::AbstractDict)
    Dict{String,Any}(string(k) => flatten_arrays(v) for (k, v) in pairs(x))
end
function flatten_arrays(x::AbstractArray{<:Any, N}) where {N}
    N <= 1 ? map(flatten_arrays, x) : map(flatten_arrays, vec(x))
end

function save_json(todict_function, filename::AbstractString, scfres::NamedTuple;
                   save_ψ=false, extra_data=Dict{String,Any}(), save_ρ=false, kwargs...)
    if save_ψ
        @warn "Saving the bloch waves (save_ψ=true) not supported with the json format."
    end
    data = todict_function(scfres; save_ψ, save_ρ)
    for (k, v) in pairs(extra_data)
        data[k] = v
    end
    if mpi_master(scfres.basis.comm_kpts)
        open(filename * ".new", "w") do io
            JSON.json(io, flatten_arrays(data))
        end
        mv(filename * ".new", filename; force=true)
    end
    DFTK.mpi_barrier(scfres.basis.comm_kpts)
    nothing
end
function DFTK.save_scfres(::Val{:json}, filename::AbstractString, scfres::NamedTuple; kwargs...)
    save_json(DFTK.scfres_to_dict, filename, scfres; kwargs...)
end
function DFTK.save_bands(::Val{:json}, filename::AbstractString, band_data::NamedTuple; kwargs...)
    save_json(DFTK.band_data_to_dict, filename, band_data; kwargs...)
end

function DFTK.save_debugdump(::Val{:json}, prefix::AbstractString, comm::MPI.Comm,
                             dftkalgorithm::AbstractString, data::AbstractDict)
    if mpi_master(comm) && !isempty(prefix)
        fn = "$(prefix)-$(dftkalgorithm)-$(getpid()).json"
        open(fn, "w") do io
            JSON.json(io, flatten_arrays(data))
        end
        @info "Saved debug dump for $(dftkalgorithm) to $(fn)."
    end
end

end
