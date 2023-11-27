function save_scfres_master(file::AbstractString, scfres::NamedTuple, ::Val{:jld2})
    !mpi_master() && error(
        "This function should only be called on MPI master after the k-point data has " *
        "been gathered with `gather_kpts`."
    )

    JLD2.jldopen(file, "w") do jld
        jld["__propertynames"] = propertynames(scfres)
        jld["ρ"]               = scfres.ρ
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
        :ρ     => jld["ρ"],
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
struct PlaneWaveBasisSerialisation{T <: Real, Arch <: AbstractArchitecture}
    model::Model{T,T}
    Ecut::T
    variational::Bool
    kcoords::Vector{Vec3{T}}
    kweights::Vector{T}
    kgrid::Union{Nothing,Vec3{Int}}
    kshift::Union{Nothing,Vec3{T}}
    symmetries_respect_rgrid::Bool
    fft_size::Tuple{Int, Int, Int}
    architecture::Arch
end
function JLD2.writeas(::Type{PlaneWaveBasis{T,T,Arch,GT,RT,KGT}}) where {T,Arch,GT,RT,KGT}
    # The GT, GT, KGT are uniquely determined by the architecture,
    # which is stored in the basis.
    PlaneWaveBasisSerialisation{T,Arch}
end

function Base.convert(::Type{PlaneWaveBasisSerialisation{T,Arch}},
                      basis::PlaneWaveBasis{T,T,Arch}) where {T,Arch}
    PlaneWaveBasisSerialisation{T,Arch}(
        basis.model,
        basis.Ecut,
        basis.variational,
        basis.kcoords_global,
        basis.kweights_global,
        basis.kgrid,
        basis.kshift,
        basis.symmetries_respect_rgrid,
        basis.fft_size,
        basis.architecture
    )
end

function Base.convert(::Type{PlaneWaveBasis{T,T,Arch,GT,RT,KGT}},
                      serial::PlaneWaveBasisSerialisation{T,Arch}) where {T,Arch,GT,RT,KGT}
    PlaneWaveBasis(serial.model, serial.Ecut, serial.kcoords, serial.kweights;
                   serial.fft_size,
                   serial.kgrid,
                   serial.kshift,
                   serial.symmetries_respect_rgrid,
                   serial.variational,
                   architecture=serial.architecture)
end
