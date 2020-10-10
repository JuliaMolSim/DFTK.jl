import JLD2

"""
Adds simplistic checkpointing to a DFTK self-consistent field calculation.
"""
function ScfCheckpoint(filename="scfcheckpoint.jld2"; cleanup=true, overwrite=false)
    function callback(info)
        if info.n_iter == 1
            isfile(filename) && !overwrite && error(
                "Checkpoint $filename exists. Use 'overwrite=true' to force overwriting."
            )
        end
        if info.stage == :finalize
            # Cleanup checkpoint
            cleanup && isfile(filename) && rm(filename)
        else
            scfres = merge(info, (ρ=info.ρout, ρspin=info.ρ_spin_out))
            save_scfres(filename, scfres)
        end
    end
end


function save_basis(jld::JLD2.Group, basis::PlaneWaveBasis)
    n_kcoords = div(length(basis.kpoints), basis.model.n_spin_components)

    jld["model"]      = basis.model
    jld["Ecut"]       = basis.Ecut
    jld["kcoords"]    = getproperty.(basis.kpoints[1:n_kcoords], :coordinate)
    jld["kweights"]   = basis.kweights[1:n_kcoords]
    jld["ksymops"]    = basis.ksymops[1:n_kcoords]
    jld["fft_size"]   = basis.fft_size
    jld["symmetries"] = basis.symmetries

    jld
end

function load_basis(jld::JLD2.Group)
    PlaneWaveBasis(jld["model"], jld["Ecut"], jld["kcoords"],
                   jld["ksymops"], jld["symmetries"], fft_size=jld["fft_size"])
end

"""
    save_scfres(filename, scfres)

Save an `scfres` obtained from `self_consistent_field` to a JLD2 file.

!!! warning "No compatibility guarantees"
    No guarantees are made with respect to this function at this point.
    It may change incompatibly between DFTK versions or stop working / be removed
    in the future.
"""
function save_scfres(jld::JLD2.JLDFile, scfres)
    jld["__propertynames"] = propertynames(scfres)
    jld["ρ_real"]          = scfres.ρ.real
    jld["ρspin_real"]      = isnothing(scfres.ρspin) ? nothing : scfres.ρspin.real
    save_basis(JLD2.Group(jld, "basis"), scfres.basis)

    for symbol in propertynames(scfres)
        symbol in (:ham, :basis, :ρ, :ρspin, :energies) && continue  # special
        jld[string(symbol)] = getproperty(scfres, symbol)
    end

    jld
end
function save_scfres(file::AbstractString, scfres)
    JLD2.jldopen(file, "w") do jld
        save_scfres(jld, scfres)
    end
end


"""
    load_scfres(filename)

Load back an `scfres`, which has previously been stored with `save_scfres`.
Note the warning in `save_scfres`.
"""
Save an `scfres` obtained from `self_consistent_field` to a JLD2 file.
function load_scfres(jld::JLD2.JLDFile)
    basis   = load_basis(jld["basis"])
    scfdict = Dict{Symbol, Any}(
        :ρ     => from_real(basis, jld["ρ_real"]),
        :ρspin => nothing,
        :basis => basis
    )
    if !isnothing(jld["ρspin_real"])
        scfdict[:ρspin] = from_real(basis, jld["ρspin_real"])
    end

    for sym in jld["__propertynames"]
        sym in (:ham, :basis, :ρ, :ρspin, :energies) && continue  # special
        scfdict[sym] = jld[string(sym)]
    end
    energies, ham = energy_hamiltonian(basis, scfdict[:ψ], scfdict[:occupation];
                                       ρ=scfdict[:ρ], ρspin=scfdict[:ρspin],
                                       eigenvalues=scfdict[:eigenvalues],
                                       εF=scfdict[:εF])

    scfdict[:energies] = energies
    scfdict[:ham]      = ham
    (; (sym => scfdict[sym] for sym in keys(scfdict))...)
end
load_scfres(file::AbstractString) = jldopen(load_scfres, file, "r")
