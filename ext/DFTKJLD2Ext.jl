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
            jld["model"] = scfres.basis.model  # Save original model datastructure
            jld["kgrid"] = scfres.basis.kgrid  # Save original kgrid datastructure
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

    return (; scfdict...)

    # TODO Check custom basis for consistency with the data extracted here
    #      Make the next lines less repetitive

    if mpi_master()
        occupation = jld["occupation"]
        permutedims(occupation, reverse(1:ndims(occupation)))
        occupation = reshape(occupation, n_bands, n_kpts * n_spin)
        occupation = collect(eachrow(occupation))
    else
        occupation = nothing
    end
    scfdict[:occupation] = scatter_kpts(occupation, basis)

    if mpi_master()
        eigenvalues = jld["eigenvalues"]
        permutedims(eigenvalues, reverse(1:ndims(eigenvalues)))
        eigenvalues = reshape(eigenvalues, n_bands, n_kpts * n_spin)
        eigenvalues = collect(eachrow(eigenvalues))
    else
        eigenvalues = nothing
    end
    scfdict[:eigenvalues] = scatter_kpts(eigenvalues, basis)

    if mpi_master()
        n_G_vectors = jld["n_G_vectors"]
        permutedims(n_G_vectors, reverse(1:ndims(n_G_vectors)))
        n_G_vectors = reshape(n_G_vectors, n_kpts * n_spin)
    else
        n_G_vectors = nothing
    end
    n_G_vectors = scatter_kpts(n_G_vectors, basis)

    if mpi_master()
        ψ = jld["ψ"]
        permutedims(ψ, reverse(1:ndims(ψ)))
        ψ = reshape(ψ, n_G_max, n_bands, n_kpts * n_spin)
        ψ = collect(eachslice(ψ, dims=3))
    else
        ψ = nothing
    end
    ψ_padded = scatter_kpts(ψ, basis)
    scfdict[:ψ] = [ψk_padded[1:n_Gk, :] for (n_Gk, ψk_padded) in zip(n_G_vectors, ψ_padded)]

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
