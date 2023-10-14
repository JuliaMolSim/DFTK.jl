module DFTKJSON3Ext
using JSON3
using DFTK
using DFTK: todict

function DFTK.save_scfres_master(filename::AbstractString, scfres::NamedTuple, ::Val{:json})
    # TODO Quick and dirty solution for now.
    #      The better approach is to integrate with StructTypes.jl

    data = Dict("energies" => todict(scfres.energies), "damping" => scfres.α)
    for key in (:converged, :occupation_threshold, :εF, :eigenvalues,
                :occupation, :n_bands_converge, :n_iter, :algorithm, :norm_Δρ)
        data[string(key)] = getproperty(scfres, key)
    end

    open(filename, "w") do io
        JSON3.pretty(io, data)
    end
end

#TODO introduce `todict` functions for all sorts of datastructures (basis, ...)


function save_bands(filename::AbstractString, band_data::NamedTuple, ::Val{:json};
                    save_ψ=false)
    save_ψ && @warn "save_ψ not supported with json files"

    # TODO Quick and dirty solution for now.
    #      The better would be to have a BandData struct and use a `todict` function for it

    basis = band_data.basis
    data  = Dict{String,Any}(
        "kcoords" => gather_kpts(getproperty.(basis.kpoints, :coordinate), basis),
        "kspins"  => gather_kpts(getproperty.(basis.kpoints, :spin),       basis)
    )
    if !isnothing(band_data.εF)
        data["εF"] = band_data.εF
    end

    # MPI distributed quantities
    for key in (:eigenvalues, :eigenvalues_error, :occupation)
        if hasproperty(band_data, key) && !isnothing(getproperty(band_data, key))
            data[string(key)] = gather_kpts(getproperty(band_data, key), basis)
        end
    end

    # Diagonalisation-specific quantities (MPI-distributed)
    for key in (:n_iter, :residual_norms)
        diag_value = getproperty(band_data.diagonalization, key)
        data[string(key)] = gather_kpts(diag_value, basis)
    end

    if mpi_master()
        open(filename, "w") do io
            JSON3.pretty(io, data)
        end
    end
    MPI.Barrier(MPI.COMM_WORLD)
    nothing
end
