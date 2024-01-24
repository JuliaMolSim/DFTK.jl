module DFTKJSON3Ext
using DFTK
using JSON3
using MPI

function save_json(todict_function, filename::AbstractString, scfres::NamedTuple;
                   save_ψ=false, extra_data=Dict{String,Any}(), save_ρ=true, kwargs...)
    if save_ψ
        @warn "Saving the bloch waves (save_ψ=true) not supported with the json format."
    end
    data = todict_function(scfres; save_ψ, save_ρ)
    for (k, v) in pairs(extra_data)
        data[k] = v
    end
    if mpi_master()
        open(filename * ".new", "w") do io
            JSON3.write(io, data)
        end
        mv(filename * ".new", filename; force=true)
    end
    MPI.Barrier(MPI.COMM_WORLD)
    nothing
end
function DFTK.save_scfres(::Val{:json}, filename::AbstractString, scfres::NamedTuple; kwargs...)
    save_json(DFTK.scfres_to_dict, filename, scfres; kwargs...)
end
function DFTK.save_bands(::Val{:json}, filename::AbstractString, band_data::NamedTuple; kwargs...)
    save_json(DFTK.band_data_to_dict, filename, band_data; kwargs...)
end

end
