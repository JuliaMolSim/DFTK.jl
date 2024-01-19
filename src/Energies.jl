"""
A simple struct to contain a vector of energies, and utilities to print them in a nice format.
"""
struct Energies{T <: Number}
    keys::Vector{String}
    values::Vector{T}
end
function Energies(itr::Pair{<:AbstractString,<:Number}...)
    Energies([e[1] for e in itr], [e[2] for e in itr])
end

function Base.show(io::IO, energies::Energies)
    print(io, "Energies(total = $(energies.total))")
end
function Base.show(io::IO, ::MIME"text/plain", energies::Energies)
    println(io, "Energy breakdown (in Ha):")
    for (name, value) in energies
        @printf io "    %-20s%-10.7f\n" name value
    end
    @printf io "\n    %-20s%-15.12f" "total" energies.total
end
Base.values(energies::Energies)         = energies.values
Base.keys(energies::Energies)           = energies.keys
Base.pairs(energies::Energies)          = (k => v for (k, v) in zip(energies.keys, energies.values))
Base.iterate(energies::Energies)        = iterate(pairs(energies))
Base.iterate(energies::Energies, state) = iterate(pairs(energies), state)
Base.length(energies::Energies)         = length(energies.keys)
Base.haskey(energies::Energies, key)    = !isnothing(findfirst(isequal(key), energies.keys))

Base.propertynames(energies::Energies, ::Bool=false) = append!(Symbol.(energies.keys), :total)
function Base.getproperty(energies::Energies{T}, x::Symbol) where {T}
    if x === :total
        sum(energies.values, init=zero(T))
    elseif x === :keys || x === :values
        return getfield(energies, x)
    else
        energies[string(x)]
    end
end
function Base.getindex(energies::Energies, i)
    idx = findfirst(isequal(i), energies.keys)
    isnothing(idx) && throw(KeyError(i))
    energies.values[idx]
end

"""
Convert an `Energies` struct to a dictionary representation
"""
function todict(energies::Energies)
    ret = Dict(pairs(energies)...)
    ret["total"] = energies.total
    ret
end
