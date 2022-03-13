# Key functionality to integrate DFTK and AtomsBase

function construct_system(lattice::AbstractMatrix, atoms::Vector, positions::Vector, magnetic_moments::Vector)
    @assert length(atoms) == length(positions)
    @assert isempty(magnetic_moments) || length(magnetic_moments) == length(atoms)
    atomsbase_atoms = map(enumerate(atoms)) do (i, element)
        kwargs = Dict{Symbol, Any}(:potential => element, )
        if !isempty(magnetic_moments)
            kwargs[:magnetic_moment] = normalize_magnetic_moment(magnetic_moments[i])
        end
        if element isa ElementPsp
            kwargs[:pseudopotential] = element.psp.identifier
        end

        position = lattice * positions[i] * u"bohr"
        if atomic_symbol(element) == :X  # dummy element ... should solve this upstream
            Atom(:X, position; atomic_symbol=:X, atomic_number=0, atomic_mass=0u"u", kwargs...)
        else
            Atom(atomic_symbol(element), position; kwargs...)
        end
    end
    periodic_system(atomsbase_atoms, collect(eachcol(lattice)) * u"bohr")
end

function parse_system(system::AbstractSystem{D}) where {D}
    if !all(periodicity(system))
        error("DFTK only supports calculations with periodic boundary conditions.")
    end

    # Parse abstract system and return data required to construct model
    mtx = austrip.(reduce(hcat, bounding_box(system)))
    T = eltype(mtx)
    lattice = zeros(T, 3, 3)
    lattice[1:D, 1:D] .= mtx

    # Cache for instantiated pseudopotentials. This is done to ensure that identical
    # atoms are indistinguishable in memory, which is used in the Model constructor
    # to deduce the atom_groups.
    cached_pspelements = Dict{String, ElementPsp}()
    atoms = map(system) do atom
        if hasproperty(atom, :potential)
            atom.potential
        elseif hasproperty(atom, :pseudopotential)
            get!(cached_pspelements, atom.pseudopotential) do
                ElementPsp(atomic_symbol(atom); psp=load_psp(atom.pseudopotential))
            end
        else
            ElementCoulomb(atomic_symbol(atom))
        end
    end

    positions = map(system) do atom
        coordinate = zeros(T, 3)
        coordinate[1:D] = lattice[1:D, 1:D] \ T.(austrip.(position(atom)))
        Vec3{T}(coordinate)
    end

    magnetic_moments = map(system) do atom
        hasproperty(atom, :magnetic_moment) || return nothing
        getproperty(atom, :magnetic_moment)
    end
    if all(m -> isnothing(m) || iszero(m) || isempty(m), magnetic_moments)
        empty!(magnetic_moments)
    else
        magnetic_moments = normalize_magnetic_moment.(magnetic_moments)
    end

    # TODO Use system to determine n_electrons

    (; lattice, atoms, positions, magnetic_moments)
end


"""
Parse the passed `system` and call the inner function `f` with the result,
appending additional `args` and `kwargs`.
"""
function call_with_system(f, system::AbstractSystem, args...; kwargs...)
    @assert !(:system in keys(kwargs))
    @assert !(:magnetic_moments in keys(kwargs))
    parsed = parse_system(system)
    f(parsed.lattice, parsed.atoms, parsed.positions, args...;
      system, parsed.magnetic_moments, kwargs...)
end
