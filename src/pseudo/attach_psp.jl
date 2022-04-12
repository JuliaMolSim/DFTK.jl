# Attach pseudopotentials to an atomic system

"""
    attach_psp(system::AbstractSystem, pspmap::AbstractDict)

Return a new system with the `pseudopotential` property of all atoms set according
to the passed `pspmap`, which maps from the atomic symbol to a pseudopotential identifier.

# Examples
Select pseudopotentials for all silicon and oxygen atoms in the system.
```julia-repl
julia> attach_psp(system, Dict(:Si => "hgh/lda/si-q4", :O => "hgh/lda/o-q6")
```
"""
function attach_psp(system::AbstractSystem, pspmap::AbstractDict{Symbol,String})
    particles = map(system) do atom
        symbol = atomic_symbol(atom)

        # Pseudo or explicit potential already set
        if hasproperty(atom, :pseudopotential) || hasproperty(atom, :potential)
            Atom(; atom)
        elseif !(symbol in keys(pspmap))
            error("No pseudo identifier given for element $symbol.")
        else
            Atom(; atom, pseudopotential=pspmap[symbol])
        end
    end
    FlexibleSystem(system; particles)
end


function compute_pspmap(symbols::AbstractVector{Symbol}; core=:fullcore, kwargs...)
    pspmap = map(unique(sort(symbols))) do symbol
        list = list_psp(symbol; core, kwargs...)
        if length(list) != 1
            error("Parameters passed do not uniquely identify a PSP file for element $symbol.")
        end
        symbol => list[1].identifier
    end
    Dict(pspmap...)
end
function compute_pspmap(system::AbstractSystem; kwargs...)
    compute_pspmap(atomic_symbol(system); kwargs...)
end


"""
    attach_psp(system::AbstractSystem; family=..., functional=..., core=...)

For each atom look up a pseudopotential in the library using `list_psp`, which matches the
passed parameters and store its identifier in the `pseudopotential` property of all atoms.

# Examples
Select HGH pseudopotentials for LDA XC functionals for all atoms in the system.
```julia-repl
julia> attach_psp(system; family="hgh", functional="lda")
```
"""
function attach_psp(system::AbstractSystem; kwargs...)
    attach_psp(system, compute_pspmap(system; kwargs...))
end
