# Attach pseudopotentials to an atomic system

"""
    attach_psp(system::AbstractSystem, pspmap::AbstractDict{Symbol,String})
    attach_psp(system::AbstractSystem; psps::String...)

Return a new system with the `pseudopotential` property of all atoms set according
to the passed `pspmap`, which maps from the atomic symbol to a pseudopotential identifier.
Alternatively the mapping from atomic symbol to pseudopotential identifier
can also be passed as keyword arguments.
An empty string can be used to denote elements where the full Coulomb potential should
be employed.

# Examples
Select pseudopotentials for all silicon and oxygen atoms in the system.
```julia-repl
julia> attach_psp(system, Dict(:Si => "hgh/lda/si-q4", :O => "hgh/lda/o-q6")
```

Same thing but using the kwargs syntax:
```julia-repl
julia> attach_psp(system, Si="hgh/lda/si-q4", O="hgh/lda/o-q6")
```
"""
function attach_psp(system::AbstractSystem, pspmap::AbstractDict{Symbol,String})
    particles = map(system) do atom
        symbol = element_symbol(atom)

        # Pseudo or explicit potential already set
        if haskey(atom, :pseudopotential) && !isempty(atom[:pseudopotential])
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
    attach_psp(system, Dict{Symbol,String}(pspmap...))
end
