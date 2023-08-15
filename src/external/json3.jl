function save_scfres_master(filename::AbstractString, scfres::NamedTuple, ::Val{:json})
    # TODO Quick and dirty solution for now.
    #      The better approach is to integrate with StructTypes.jl
    energies = Dict(k => getproperty(scfres.energies, k)
                    for k in propertynames(scfres.energies))

    data = Dict("energies" => energies, "damping" => scfres.α)
    for key in (:converged, :occupation_threshold, :εF, :eigenvalues,
                :occupation, :n_bands_converge, :n_iter, :algorithm, :norm_Δρ)
        data[string(key)] = getproperty(scfres, key)
    end

    open(filename, "w") do io
        JSON3.pretty(io, data)
    end
end
