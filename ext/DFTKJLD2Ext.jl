module DFTKJLD2Ext
using DFTK
using JLD2
using MPI

DFTK.make_subdict!(jld::Union{JLD2.Group,JLD2.JLDFile}, name::AbstractString) = JLD2.Group(jld, name)

function save_jld2(to_dict_function!, file::AbstractString, scfres::NamedTuple;
                   save_ψ=true, save_ρ=true, extra_data=Dict{String,Any}(), compress=false)
    if mpi_master()
        JLD2.jldopen(file * ".new", "w"; compress) do jld
            to_dict_function!(jld, scfres; save_ψ, save_ρ)
            for (k, v) in pairs(extra_data)
                jld[k] = v
            end

            # Save some original datastructures (where JLD2 can easily preserve more)
            jld["model"] = scfres.basis.model  # Save original model datastructure
            delete!(jld, "kgrid")
            jld["kgrid"] = scfres.basis.kgrid  # Save original kgrid datastructure
        end
        mv(file * ".new", file; force=true)
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


function DFTK.load_scfres(::Val{:jld2}, filename::AbstractString, basis=nothing; kwargs...)
    if mpi_master()
        scfres = JLD2.jldopen(filename, "r") do jld
            load_scfres_jld2(jld, basis; kwargs...)
        end
    else
        scfres = load_scfres_jld2(nothing, basis; kwargs...)
    end
    MPI.Barrier(MPI.COMM_WORLD)
    scfres
end
function load_scfres_jld2(jld, basis; skip_hamiltonian, strict)
    consistent_kpts = true
    if isnothing(basis)
        basis = load_basis(jld)
    else
        errormsg = ""
        if mpi_master()
            # Check custom basis for consistency with the data extracted here
            if !(basis.architecture isa DFTK.CPU)  # TODO Else need to put things on the GPU
                errormsg = "Only CPU architectures supported for now."
            end
            if jld["fft_size"] != basis.fft_size
                errormsg = ("Mismatch in fft_size between file ($(jld["fft_size"])) " *
                            "and supplied basis ($(basis.fft_size))")
            end
            if jld["n_kpoints"] != length(basis.kcoords_global)
                consistent_kpts = false
                strict && (errormsg = "Mismatch in number of k-points between file " *
                                      "($(jld["n_kpoints"])) and supplied basis " *
                                      "($(length(basis.kcoords_global)))")
            end
            if jld["n_spin_components"] != basis.model.n_spin_components
                consistent_kpts = false
                errormsg = ("Mismatch in number of spin components between file " *
                            "($(jld["n_spin_components"])) and supplied basis " *
                            "($(basis.model.n_spin_components))")
            end
        end
        errormsg = MPI.bcast(errormsg, 0, MPI.COMM_WORLD)
        isempty(errormsg) || error(errormsg)
    end

    propmap = Dict(:damping_value => :α, )  # compatibility mapping
    if mpi_master()
        # Setup default energies
        e_keys   = filter!(!isequal("total"), collect(keys(jld["energies"])))
        e_values = [jld["energies"][k] for k in e_keys]

        scfdict = Dict{Symbol, Any}(
            :εF       => get(jld, "εF", nothing),
            :ρ        => get(jld, "ρ", nothing),
            :ψ        => nothing,
            :energies => DFTK.Energies(e_keys, e_values),
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
            data = jld[key]
            n = ndims(data)
            data = reshape(data, size(data)[1:n-2]..., size(data, n-1) * size(data, n))
        else
            data = nothing
        end
        DFTK.scatter_kpts_block(basis, data)
    end

    # TODO Could also reconstruct diagonalization data structure

    if consistent_kpts
        scfdict[:eigenvalues] = reshape_and_scatter(jld, "eigenvalues")
        scfdict[:occupation]  = reshape_and_scatter(jld, "occupation")
    end

    has_ψ = mpi_master() ? (consistent_kpts && haskey(jld, "ψ")) : nothing
    has_ψ = MPI.bcast(has_ψ, 0, MPI.COMM_WORLD)
    if has_ψ
        n_G_vectors = reshape_and_scatter(jld, "kpt_n_G_vectors")
        basisok = all(n_G_vectors[ik] == length(DFTK.G_vectors(basis, kpt))
                      for (ik, kpt) in enumerate(basis.kpoints))
        basisok = DFTK.mpi_min(basisok, basis.comm_kpts)
        if basisok
            ψ_padded = reshape_and_scatter(jld, "ψ")
            scfdict[:ψ] = DFTK.unblockify_ψ(ψ_padded, n_G_vectors)
        elseif strict
            error("Mismatch in number of G-vectors per k-point.")
        end
    end

    if !skip_hamiltonian && has_ψ && !isnothing(scfdict[:ρ])
        energies, ham = DFTK.energy_hamiltonian(basis, scfdict[:ψ], scfdict[:occupation];
                                                ρ=scfdict[:ρ],
                                                eigenvalues=scfdict[:eigenvalues],
                                                εF=scfdict[:εF])
        scfdict[:energies] = energies
        scfdict[:ham]      = ham
    end

    MPI.Barrier(MPI.COMM_WORLD)
    (; scfdict...)
end

end
