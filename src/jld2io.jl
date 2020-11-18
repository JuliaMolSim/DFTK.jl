import JLD2

"""
Adds simplistic checkpointing to a DFTK self-consistent field calculation.
"""
function ScfSaveCheckpoints(filename="dftk_scf_checkpoint.jld2"; keep=false, overwrite=false)
    # TODO Save only every 30 minutes or so
    function callback(info)
        if info.n_iter == 1
            isfile(filename) && !overwrite && error(
                "Checkpoint $filename exists. Use 'overwrite=true' to force overwriting."
            )
        end
        if info.stage == :finalize
            !keep && isfile(filename) && rm(filename)  # Cleanup checkpoint
        else
            scfres = (; (k => v for (k, v) in pairs(info) if !startswith(string(k), "ρ"))...)
            scfres = merge(scfres, (ρ=info.ρout, ρspin=info.ρ_spin_out))
            save_scfres(filename, scfres)
        end
        info
    end
end


struct PlaneWaveBasisSerialisation{T <: Real}
    model
    Ecut
    kcoords
    kweights
    ksymops
    fft_size
    symmetries
end

function JLD2.writeas(::Type{PlaneWaveBasisSerialisation{T}}) where {T}
    PlaneWaveBasisSerialisation{T}
end

function Base.convert(::Type{PlaneWaveBasisSerialisation{T}},
                      basis::PlaneWaveBasis{T}) where {T}
    if mpi_nprocs() > 1
        error("JLD2 serialisation for PlaneWaveBasis only implemented for non-MPI calculations for now.")
    end
    n_kcoords = div(length(basis.kpoints), basis.model.n_spin_components)

    PlaneWaveBasisSerialisation{T}(
        basis.model,
        basis.Ecut,
        getproperty.(basis.kpoints[1:n_kcoords], :coordinate),
        basis.kweights[1:n_kcoords],
        basis.ksymops[1:n_kcoords],
        basis.fft_size,
        basis.symmetries
    )
end

function Base.convert(::Type{PlaneWaveBasis{T}},
                      serialised::PlaneWaveBasisSerialisation{T}) where {T}
    PlaneWaveBasis{T}(serialised.model, serialised.Ecut, serialised.kcoords,
                      serialised.ksymops, serialised.symmetries,
                      fft_size=serialised.fft_size)
end


function save_scfres(jld::JLD2.JLDFile, scfres::NamedTuple)
    jld["__propertynames"] = propertynames(scfres)
    jld["ρ_real"]          = scfres.ρ.real
    jld["ρspin_real"]      = isnothing(scfres.ρspin) ? nothing : scfres.ρspin.real
    jld["basis"]           = scfres.basis

    for symbol in propertynames(scfres)
        symbol in (:ham, :basis, :ρ, :ρspin, :energies) && continue  # special
        jld[string(symbol)] = getproperty(scfres, symbol)
    end

    jld
end
function save_scfres(file::AbstractString, scfres::NamedTuple)
    JLD2.jldopen(file, "w") do jld
        save_scfres(jld, scfres)
    end
end


function load_scfres(jld::JLD2.JLDFile)
    basis = jld["basis"]
    scfdict = Dict{Symbol, Any}(
        :ρ     => from_real(basis, jld["ρ_real"]),
        :ρspin => nothing,
        :basis => basis
    )
    if !isnothing(jld["ρspin_real"])
        scfdict[:ρspin] = from_real(basis, jld["ρspin_real"])
    end

    kpt_properties = (:ψ, :occupation, :eigenvalues)  # Need splitting over MPI processes
    for sym in kpt_properties
        scfdict[sym] = jld[string(sym)][basis.krange_thisproc]
    end
    for sym in jld["__propertynames"]
        sym in (:ham, :basis, :ρ, :ρspin, :energies) && continue  # special
        sym in kpt_properties && continue
        scfdict[sym] = jld[string(sym)]
    end

    energies, ham = energy_hamiltonian(basis, scfdict[:ψ], scfdict[:occupation];
                                       ρ=scfdict[:ρ], ρspin=scfdict[:ρspin],
                                       eigenvalues=scfdict[:eigenvalues],
                                       εF=scfdict[:εF])

    scfdict[:energies] = energies
    scfdict[:ham]      = ham
    (; (sym => scfdict[sym] for sym in jld["__propertynames"])...)
end
load_scfres(file::AbstractString) = JLD2.jldopen(load_scfres, file, "r")
