using AtomsBase
# Key functionality to integrate DFTK and AtomsBase

function parse_system(system::AbstractSystem{D},
                      pseudopotentials::AbstractVector,
                      pseudofamily::Union{PseudoFamily,Nothing}=nothing) where {D}
    # pseudofamily !== nothing marks a case where all pseudopotentials are from
    # exactly the same PseudoFamily. This helps downstream to make some smart
    # default choices about Ecut etc.
    #
    if !all(periodicity(system))
        error("DFTK only supports calculations with periodic boundary conditions.")
    end
    if length(system) != length(pseudopotentials)
        error("Length of pseudopotentials vector needs to agree with number of atoms.")
    end

    # Strip unit from a quantity type (like ustrip for types)
    strip_unit(::Type{<:Quantity{T}}) where {T} = T
  
    # Parse abstract system and return data required to construct model
    mtx = austrip.(stack(cell_vectors(system)))
    T = reduce(promote_type, strip_unit.(eltype.(position(system, :))); init=eltype(mtx))
    lattice = zeros(T, 3, 3)
    lattice[1:D, 1:D] .= mtx

    atoms = map(system, pseudopotentials) do atom, psp
        if hasproperty(atom, :pseudopotential)
            @warn("The pseudopotential atom property is no longer respected. " *
                  "Please use the `pseudopotential` keyword argument to the " *
                  "model constructors.")
        end
        # If psp === nothing, this will make an ElementCoulomb
        ElementPsp(species(atom), psp, pseudofamily; mass=AtomsBase.mass(atom))
    end

    positions = map(system) do atom
        coordinate = zeros(T, 3)
        coordinate[1:D] = lattice[1:D, 1:D] \ T.(austrip.(position(atom)))
        Vec3{T}(coordinate)
    end

    magnetic_moments = map(system) do atom
        get(atom, :magnetic_moment, nothing)
    end
    if all(m -> isnothing(m) || iszero(m) || isempty(m), magnetic_moments)
        empty!(magnetic_moments)
    else
        magnetic_moments = normalize_magnetic_moment.(magnetic_moments)
    end

    sum_atomic_charge = sum(atom -> get(atom, :charge, 0.0u"e_au"), system; init=0.0u"e_au")
    if abs(sum_atomic_charge) > 1e-6u"e_au"
        error("Charged systems not yet supported in DFTK.")
    end

    if !iszero(get(system, :charge, 0.0u"e_au"))
        error("Charged systems not yet supported in DFTK.")
    end
    for k in (:multiplicity, )
        if haskey(system, k)
            @warn "System property $k not supported and ignored in DFTK."
        end
    end

    (; lattice, atoms, positions, magnetic_moments)
end
function parse_system(system::AbstractSystem,
                      family::AbstractDict{Symbol,<:AbstractString})
    parse_system(system, load_psp(family, system), nothing)
end
function parse_system(system::AbstractSystem, family::PseudoFamily)
    parse_system(system, load_psp(family, system), family)
end


# Extra methods to AtomsBase functions for DFTK data structures
"""
    atomic_system(model::DFTK.Model, magnetic_moments=[])
    atomic_system(lattice, atoms, positions, magnetic_moments=[])

Construct an AtomsBase atomic system from a DFTK `model` and associated magnetic moments
or from the usual `lattice`, `atoms` and `positions` list used in DFTK plus magnetic moments.
"""
function AtomsBase.atomic_system(lattice::AbstractMatrix{<:Number},
                                 atoms::Vector{<:Element},
                                 positions::AbstractVector,
                                 magnetic_moments::AbstractVector=[])
    lattice = austrip.(lattice)

    @assert length(atoms) == length(positions)
    @assert isempty(magnetic_moments) || length(magnetic_moments) == length(atoms)
    atomsbase_atoms = map(enumerate(atoms)) do (i, atom)
        kwargs = Dict{Symbol, Any}()
        if !isempty(magnetic_moments)
            magmom = normalize_magnetic_moment(magnetic_moments[i])
            if iszero(magmom[1]) && iszero(magmom[2])
                kwargs[:magnetic_moment] = magmom[3]
            else
                kwargs[:magnetic_moment] = magmom
            end
        end

        Atom(species(atom), lattice * positions[i] * u"bohr";
             mass=AtomsBase.mass(atom), kwargs...)
    end
    periodic_system(atomsbase_atoms, collect(eachcol(lattice)) * u"bohr")
end
function AtomsBase.atomic_system(model::Model, magnetic_moments=[])
    atomic_system(model.lattice, model.atoms, model.positions, magnetic_moments)
end


"""
    periodic_system(model::DFTK.Model, magnetic_moments=[])
    periodic_system(lattice, atoms, positions, magnetic_moments=[])

Construct an AtomsBase atomic system from a DFTK `model` and associated magnetic moments
or from the usual `lattice`, `atoms` and `positions` list used in DFTK plus magnetic moments.
"""
function AtomsBase.periodic_system(model::Model, magnetic_moments=[])
    atomic_system(model, magnetic_moments)
end
function AtomsBase.periodic_system(lattice::AbstractMatrix{<:Number},
                                   atoms::Vector{<:Element},
                                   positions::AbstractVector{<:AbstractVector},
                                   magnetic_moments::AbstractVector=[])
    atomic_system(lattice, atoms, positions, magnetic_moments)
end

function AtomsBase.chemical_formula(model::Model)
    chemical_formula(element_symbol.(species.(model.atoms)))
end
