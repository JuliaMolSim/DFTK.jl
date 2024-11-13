using AtomsBase
using PseudoPotentialData
# Key functionality to integrate DFTK and AtomsBase

function parse_system(system::AbstractSystem{D},
                      pseudopotentials::AbstractVector) where {D}
    if !all(periodicity(system))
        error("DFTK only supports calculations with periodic boundary conditions.")
    end
    if length(system) != length(pseudopotentials)
        error("Length of pseudopotentials vector needs to agree with number of atoms.")
    end

    # Parse abstract system and return data required to construct model
    mtx = austrip.(stack(bounding_box(system)))
    T = eltype(mtx)
    lattice = zeros(T, 3, 3)
    lattice[1:D, 1:D] .= mtx

    atoms = map(system, pseudopotentials) do atom, psp
        if hasproperty(atom, :pseudopotential)
            @warn("The pseudopotential atom property is no longer respected. " *
                  "Please use the `pseudopotential` keyword argument to the " *
                  "model constructors.")
        end
        mass = atomic_mass(atom)
        if isnothing(psp)
            ElementCoulomb(atomic_number(atom); mass)
        else
            ElementPsp(atomic_number(atom); psp, mass)
        end
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
function parse_system(system::AbstractSystem, family::PseudoFamily)
    parse_system(system, load_psp(family, system))
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
    atomsbase_atoms = map(enumerate(atoms)) do (i, element)
        kwargs = Dict{Symbol, Any}()
        if !isempty(magnetic_moments)
            magmom = normalize_magnetic_moment(magnetic_moments[i])
            if iszero(magmom[1]) && iszero(magmom[2])
                kwargs[:magnetic_moment] = magmom[3]
            else
                kwargs[:magnetic_moment] = magmom
            end
        end

        position = lattice * positions[i] * u"bohr"
        if atomic_symbol(element) == :X  # dummy element ... should solve this upstream
            Atom(:X, position; atomic_symbol=:X, atomic_number=0, atomic_mass=0u"u", kwargs...)
        else
            Atom(atomic_symbol(element), position; atomic_mass=atomic_mass(element),
                 kwargs...)
        end
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

AtomsBase.chemical_formula(model::Model) = chemical_formula(atomic_symbol.(model.atoms))
