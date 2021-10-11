
function ScfSaveCheckpoints(filename="dftk_scf_checkpoint.jld2"; keep=false, overwrite=false)
    # TODO Save only every 30 minutes or so
    function callback(info)
        if info.n_iter == 1
            isfile(filename) && !overwrite && error(
                "Checkpoint $filename exists. Use 'overwrite=true' to force overwriting."
            )
        end
        if info.stage == :finalize
            if mpi_master() && !keep
                isfile(filename) && rm(filename)  # Cleanup checkpoint
            end
        else
            scfres = (; (k => v for (k, v) in pairs(info) if !startswith(string(k), "ρ"))...)
            scfres = merge(scfres, (; ρ=info.ρout))
            save_scfres(filename, scfres)
        end
        info
    end
end


function save_scfres_master(file::AbstractString, scfres::NamedTuple, ::Val{:jld2})
    !mpi_master() && error(
        "This function should only be called on MPI master after the k-point data has " *
        "been gathered with `gather_kpts`."
    )

    JLD2.jldopen(file, "w") do jld
        jld["__propertynames"] = propertynames(scfres)
        jld["ρ_real"]          = scfres.ρ
        jld["basis"]           = scfres.basis

        for symbol in propertynames(scfres)
            symbol in (:ham, :basis, :ρ, :energies) && continue  # special
            jld[string(symbol)] = getproperty(scfres, symbol)
        end

        jld
    end
end


function load_scfres(jld::JLD2.JLDFile)
    basis = jld["basis"]
    scfdict = Dict{Symbol, Any}(
        :ρ     => jld["ρ_real"],
        :basis => basis
    )

    kpt_properties = (:ψ, :occupation, :eigenvalues)  # Need splitting over MPI processes
    for sym in kpt_properties
        scfdict[sym] = jld[string(sym)][basis.krange_thisproc]
    end
    for sym in jld["__propertynames"]
        sym in (:ham, :basis, :ρ, :energies) && continue  # special
        sym in kpt_properties && continue
        scfdict[sym] = jld[string(sym)]
    end

    energies, ham = energy_hamiltonian(basis, scfdict[:ψ], scfdict[:occupation];
                                       ρ=scfdict[:ρ],
                                       eigenvalues=scfdict[:eigenvalues],
                                       εF=scfdict[:εF])

    scfdict[:energies] = energies
    scfdict[:ham]      = ham
    (; (sym => scfdict[sym] for sym in jld["__propertynames"])...)
end
load_scfres(file::AbstractString) = JLD2.jldopen(load_scfres, file, "r")


#
# Custom serialisations
#
struct PlaneWaveBasisSerialisation{T <: Real}
    model::Model{T}
    Ecut::T
    variational::Bool
    kcoords::Vector{Vec3{T}}
    ksymops::Vector{Vector{SymOp}}
    kgrid::Union{Nothing,Vec3{Int}}
    kshift::Union{Nothing,Vec3{T}}
    fft_size::Tuple{Int, Int, Int}
    symmetries::Vector{SymOp}
end
JLD2.writeas(::Type{PlaneWaveBasis{T}}) where {T} = PlaneWaveBasisSerialisation{T}

function Base.convert(::Type{PlaneWaveBasisSerialisation{T}},
                      basis::PlaneWaveBasis{T}) where {T}
    # Notice: This function is only meaningful on a basis which has gathered
    # all k-point information locally. So before using this in MPI-distributed calculations,
    # ensure to call `mpi_kgather` on the PlaneWaveBasis object.

    # Number of distinct k-point coordinates
    n_kcoords = div(length(basis.kpoints), basis.model.n_spin_components)
    PlaneWaveBasisSerialisation{T}(
        basis.model,
        basis.Ecut,
        basis.variational,
        getproperty.(basis.kpoints[1:n_kcoords], :coordinate),
        basis.ksymops[1:n_kcoords],
        basis.kgrid,
        basis.kshift,
        basis.fft_size,
        basis.symmetries
    )
end

function Base.convert(::Type{PlaneWaveBasis{T}},
                      serial::PlaneWaveBasisSerialisation{T}) where {T}
    PlaneWaveBasis(serial.model, serial.Ecut, serial.kcoords,
                   serial.ksymops, serial.symmetries;
                   fft_size=serial.fft_size,
                   kgrid=serial.kgrid, kshift=serial.kshift,
                   variational=serial.variational)
end
