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
    n_bands   = length(band_data.eigenvalues[1])
    n_kpoints = length(basis.kcoords_global)
    n_spin    = basis.model.n_spin_components

    data  = Dict{String,Any}(
        "n_kpoints" => n_kpoints,
        "n_spin"    => n_spin,
        "n_bands"   => n_bands,
        "kcoords"   => basis.kcoords_global,
    )
    if !isnothing(band_data.εF)
        data["εF"] = band_data.εF
    end

    # Gather MPI distributed on the first processor and reshape into an
    # (n_spin, n_kpoints, n_bands) arrays
    function gather_and_reshape(data, shape)
        value = gather_kpts(data, basis)
        if mpi_master()
            value = reshape(reduce(hcat, value), reverse(shape)...)
            permutedims(value, reverse(1:length(shape)))
        else
            nothing
        end
    end
    for key in (:eigenvalues, :eigenvalues_error, :occupation)
        if hasproperty(band_data, key) && !isnothing(getproperty(band_data, key))
            data[string(key)] = gather_and_reshape(getproperty(band_data, key),
                                                   (n_spin, n_kpoints, n_bands))
        end
    end
    data["n_iter"]         = gather_and_reshape(band_data.diagonalization.n_iter,
                                                (n_spin, n_kpoints))
    data["residual_norms"] = gather_and_reshape(band_data.diagonalization.residual_norms,
                                                (n_spin, n_kpoints, n_bands))

    if mpi_master()
        open(filename, "w") do io
            JSON3.pretty(io, data)
        end
    end
    MPI.Barrier(MPI.COMM_WORLD)
    nothing
end
