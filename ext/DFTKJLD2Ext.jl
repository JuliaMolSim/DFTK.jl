module DFTKJLD2Ext
using DFTK
using JLD2
using MPI

DFTK.make_subdict!(jld::Union{JLD2.Group,JLD2.JLDFile}, name::AbstractString) = JLD2.Group(jld, name)

function save_jld2(to_dict_function!, file::AbstractString, scfres::NamedTuple;
                   save_ψ=true, extra_data=Dict{String,Any}())
    if mpi_master()
        JLD2.jldopen(file, "w") do jld
            to_dict_function!(jld, scfres; save_ψ)
            delete!(jld, "kgrid")
            jld["kgrid"] = scfres.basis.kgrid  # Save original kgrid datastructure
            jld["model"] = scfres.basis.model  # Save original model datastructure
            for (k, v) in pairs(extra_data)
                jld[k] = v
            end
        end
    else
        dummy = Dict{String,Any}()
        to_dict_function!(dummy, scfres; save_ψ)
    end
    MPI.Barrier(MPI.COMM_WORLD)
    nothing
end
function DFTK.save_scfres(::Val{:jld2}, file::AbstractString, scfres::NamedTuple; kwargs...)
    save_jld2(DFTK.scfres_to_dict!, file, scfres; kwargs...)
end
function DFTK.save_bands(::Val{:jld2}, file::AbstractString, band_data::NamedTuple; kwargs...)
    save_jld2(DFTK.band_data_to_dict!, file, band_data; kwargs...)
end

function load_basis(jld)
    if mpi_master()
        basis_args = (jld["model"],
                      jld["Ecut"],
                      jld["fft_size"],
                      jld["variational"],
                      jld["kgrid"],
                      jld["symmetries_respect_rgrid"],
                      jld["use_symmetries_for_kpoint_reduction"])
    else
        basis_args = nothing
    end
    basis_args = MPI.bcast(basis_args, 0, MPI.COMM_WORLD)
    PlaneWaveBasis(basis_args..., MPI.COMM_WORLD, DFTK.CPU())
end


function DFTK.load_scfres(::Val{:jld2}, filename::AbstractString, basis=nothing;
                          skip_hamiltonian=false)
    if mpi_master()
        scfres = JLD2.jldopen(filename, "r") do jld
            load_scfres_jld2(jld, basis; skip_hamiltonian)
        end
    else
        scfres = load_scfres_jld2(nothing, basis; skip_hamiltonian)
    end
    MPI.Barrier(MPI.COMM_WORLD)
    scfres
end
function load_scfres_jld2(jld, basis; skip_hamiltonian)
    if isnothing(basis)
        basis = load_basis(jld)
    end

    propmap = Dict(:damping => :α, )  # compatibility mapping
    if mpi_master()
        scfdict = Dict{Symbol, Any}(
            :εF => jld["εF"],
            :ρ  => jld["ρ"],
        )
        for key in jld["scfres_extra_keys"]
            scfdict[get(propmap, Symbol(key), Symbol(key))] = jld[key]
        end
    else
        scfdict = nothing
    end
    scfdict = MPI.bcast(scfdict, 0, MPI.COMM_WORLD)

    function reshape_and_scatter(data)
        if mpi_master()
            n = ndims(data)
            value = reshape(data, size(data)[1:n-2]..., size(data, n-1) * size(data, n))
            if ndims(value) > 1
                value = collect(eachslice(value, dims=n-1))
            end
        else
            value = nothing
        end
        DFTK.scatter_kpts(value, basis)
    end

    # TODO Check if this Array.( ... ) is really needed. My suspicion is
    #      otherwise one gets a view here.
    scfdict[:eigenvalues] = Array.(reshape_and_scatter(jld["eigenvalues"]))
    scfdict[:occupation]  = Array.(reshape_and_scatter(jld["occupation"]))

    n_G_vectors = reshape_and_scatter(jld["kpt_n_G_vectors"])
    ψ_padded = reshape_and_scatter(jld["ψ"])
    scfdict[:ψ] = map(n_G_vectors, ψ_padded) do n_Gk, ψk_padded
        ψk_padded[1:n_Gk, :]
    end

    #
    # TODO Put on the GPU if needed
    #
    # TODO Check custom basis for consistency with the data extracted here
    #      Make the next lines less repetitive
    #

    if !skip_hamiltonian
        energies, ham = DFTK.energy_hamiltonian(basis, scfdict[:ψ], scfdict[:occupation];
                                                ρ=scfdict[:ρ],
                                                eigenvalues=scfdict[:eigenvalues],
                                                εF=scfdict[:εF])
        scfdict[:energies] = energies
        scfdict[:ham]      = ham
    else
        # TODO reconstruct energies from the data we have in the jld
    end

    MPI.Barrier(MPI.COMM_WORLD)
    (; scfdict...)
end

end
