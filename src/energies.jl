using OrderedCollections
"""
A simple struct to contain a vector of energies, and utilities to print them in a nice format.
"""
struct Energies{T <: Number}
    # energies["TermName"]
    # parametrization on T acts as a nice check that all terms return correct type
    energies::OrderedDict{String, T}
end

function Base.show(io::IO, E::Energies)
    energies = E.energies
    println("\nEnergy breakdown:")
    for (name, value) in E.energies
        @printf "    %-20s%-10.7f\n" string(name) value
    end
    @printf "\n    %-20s%-15.12f\n" "total" sum(E)
end
Base.getindex(E::Energies, i) = E.energies[i]

import Base.sum
Base.sum(E::Energies) = sum(values(E.energies))

function Energies(basis::PlaneWaveBasis, energies::Vector)
    # nameof is there to get rid of parametric types
    Energies(OrderedDict([string(nameof(typeof(basis.model.term_types[it]))) => energies[it]
                          for it = 1:length(basis.model.term_types)]...))
end
