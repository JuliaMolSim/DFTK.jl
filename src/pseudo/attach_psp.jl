# Attach pseudopotentials to an atomic system

"""
    attach_psp(system::AbstractSystem, pspmap::AbstractDict{Symbol,AbstractPsP})
    attach_psp(system::AbstractSystem; psps::AbstractPsP...)

Return a new system with the `pseudopotential` property of all atoms set according
to the passed `pspmap`, which maps from the atomic symbol to a pseudopotential identifier.
Alternatively the mapping from atomic symbol to pseudopotential identifier
can also be passed as keyword arguments.
An empty string can be used to denote elements where the full Coulomb potential should
be employed.

# Examples
Select pseudopotentials for all silicon and oxygen atoms in the system.
```julia-repl
julia> attach_psp(system,
                  Dict(:Si => PseudoPotentialIO.load_psp("hgh_lda_hgh", "si-q4.hgh"),
                       :O => PseudoPotentialIO.load_psp("hgh_lda_hgh", "o-q6.hgh"))
```

Same thing but using the kwargs syntax:
```julia-repl
julia> attach_psp(system, Si=PseudoPotentialIO.load_psp("hgh_lda_hgh", "si-q4.hgh"),
                  O=PseudoPotentialIO.load_psp("hgh_lda_hgh", "o-q6.hgh"))
```
"""
function attach_psp(system::AbstractSystem, pspmap::AbstractDict{Symbol,T}) where {T<:PseudoPotentialIO.AbstractPsP}
    particles = map(system) do atom
        symbol = atomic_symbol(atom)

        # Pseudo or explicit potential already set
        if haskey(atom, :pseudopotential) && !isnothing(atom[:pseudopotential])
            Atom(; pairs(atom)...)
        elseif !(symbol in keys(pspmap))
            error("No pseudo identifier given for element $symbol.")
        else
            Atom(; pairs(atom)..., pseudopotential=pspmap[symbol])
        end
    end
    FlexibleSystem(system; particles)
end
function attach_psp(system::AbstractSystem; pspmap...)
    attach_psp(system, Dict{Symbol,AbstractPsP}(pspmap...))
end
