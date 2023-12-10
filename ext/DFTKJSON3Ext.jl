module DFTKJSON3Ext
using JSON3
using DFTK
using DFTK: todict

function DFTK.save_scfres_master(filename::AbstractString, scfres::NamedTuple, ::Val{:json})
    # TODO Quick and dirty solution for now.
    #      The better approach is to integrate with StructTypes.jl
    # Also should probably be merged with save_bands eventually
    # once the treatment of MPI distributed data is uniform.

    data = Dict("energies" => todict(scfres.energies), "damping" => scfres.α)
    for key in (:converged, :occupation_threshold, :εF, :eigenvalues,
                :occupation, :n_bands_converge, :n_iter, :algorithm, :norm_Δρ)
        data[string(key)] = getproperty(scfres, key)
    end

    open(filename, "w") do io
        JSON3.pretty(io, data)
    end
end

function save_bands(filename::AbstractString, band_data::NamedTuple, ::Val{:json};
                    save_ψ=false)
    save_ψ && @warn "save_ψ not supported with json files"

    data = band_data_to_dict(band_data)
    if mpi_master()
        open(filename, "w") do io
            JSON3.pretty(io, data)
        end
    end
    MPI.Barrier(MPI.COMM_WORLD)
    nothing
end
