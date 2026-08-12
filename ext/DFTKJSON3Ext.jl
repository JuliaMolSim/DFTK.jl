module DFTKJSON3Ext
using DFTK
using JSON3
using MPI
using Preferences

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
            JSON3.write(io, data)
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
            JSON3.write(io, data)
        end
        @info "Saved debug dump for $(dftkalgorithm) to $(fn)."
    end
end

end
