module DFTKJLD2Ext
using DFTK
using JLD2
using MPI

DFTK.make_subdict!(jld::Union{JLD2.Group,JLD2.JLDFile}, name::AbstractString) = JLD2.Group(jld, name)

function save_jld2(to_dict_function!, file::AbstractString, scfres::NamedTuple;
                   save_ψ=true, extra_data=Dict{String,Any}(), compress=false)
    if mpi_master()
        JLD2.jldopen(file, "w"; compress) do jld
            to_dict_function!(jld, scfres; save_ψ)
            for (k, v) in pairs(extra_data)
                jld[k] = v
            end

            # Save some original datastructures (where JLD2 can easily preserve more)
            jld["model"] = scfres.basis.model  # Save original model datastructure
            delete!(jld, "kgrid")
            jld["kgrid"] = scfres.basis.kgrid  # Save original kgrid datastructure
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
    basisok = false
    if isnothing(basis)
        basis = load_basis(jld)
        basisok = true
    elseif mpi_master()
        # Check custom basis for consistency with the data extracted here
        if !(basis.architecture isa DFTK.CPU)  # TODO Else need to put things on the GPU
            error("Only CPU architectures supported for now.")
        end
        if jld["fft_size"] != basis.fft_size
            error("Mismatch in fft_size between file ($(jld["fft_size"])) " *
                  "and supplied basis ($(basis.fft_size))")
        end
        if jld["n_kpoints"] != length(basis.kcoords_global)
            error("Mismatch in number of k-points between file ($(jld["n_kpoints"])) " *
                  "and supplied basis ($(length(basis.kcoords_global)))")
        end
        if jld["n_spin_components"] != basis.model.n_spin_components
            error("Mismatch in number of spin components between file ($(jld["n_spin_components"])) " *
                  "and supplied basis ($(basis.model.n_spin_components))")
        end
        basisok = true
    end
    basisok || error("Basis not consistent")

    propmap = Dict(:damping_value => :α, )  # compatibility mapping
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
    scfdict[:basis] = basis

    function reshape_and_scatter(jld, key)
        if mpi_master()
            # TODO Performance improvement by reading the jld file in chunks ?
            #      would for sure lower the memory footprint with many k-points
            data = jld[key]
            n = ndims(data)
            value = reshape(data, size(data)[1:n-2]..., size(data, n-1) * size(data, n))
            if ndims(value) > 1
                value = [Array(s) for s in eachslice(value, dims=n-1)]
            end
        else
            value = nothing
        end
        DFTK.scatter_kpts(value, basis)
    end

    # TODO Could also reconstruct diagonalization data structure

    scfdict[:eigenvalues] = reshape_and_scatter(jld, "eigenvalues")
    scfdict[:occupation]  = reshape_and_scatter(jld, "occupation")

    has_ψ = mpi_master() ? haskey(jld, "ψ") : nothing
    has_ψ = MPI.bcast(has_ψ, 0, MPI.COMM_WORLD)
    if has_ψ
        n_G_vectors = reshape_and_scatter(jld, "kpt_n_G_vectors")

        basisok = all(n_G_vectors[ik] == length(DFTK.G_vectors(basis, kpt))
                      for (ik, kpt) in enumerate(basis.kpoints))
        basisok = DFTK.mpi_min(basisok, basis.comm_kpts)
        basisok || error("Mismatch in number of G-vectors per k-point.")

        ψ_padded = reshape_and_scatter(jld, "ψ")
        scfdict[:ψ] = map(n_G_vectors, ψ_padded) do n_Gk, ψk_padded
            ψk_padded[1:n_Gk, :]
        end
    end

    if !skip_hamiltonian && has_ψ
        energies, ham = DFTK.energy_hamiltonian(basis, scfdict[:ψ], scfdict[:occupation];
                                                ρ=scfdict[:ρ],
                                                eigenvalues=scfdict[:eigenvalues],
                                                εF=scfdict[:εF])
        scfdict[:energies] = energies
        scfdict[:ham]      = ham
    else
        terms   = filter!(!isequal("total"), collect(keys(jld["energies"])))
        values = [jld["energies"][k] for k in terms]
        scfdict[:energies] = DFTK.Energies(DFTK.OrderedDict(terms .=> values))
    end

    MPI.Barrier(MPI.COMM_WORLD)
    (; scfdict...)
end

end
