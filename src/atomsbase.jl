# Key functionality to integrate DFTK and AtomsBase

function construct_atomsbase(lattice, atoms, positions, magnetic_moments)
    error("Improve this!")
    periodic_system(Atom[], collect(eachcol(lattice)) * u"bohr")
end

function parse_atomsbase(system::AbstractSystem{D}) where {D}
    if !all(periodicity(system))
        error("DFTK only supports calculations with periodic boundary conditions.")
    end

    # Parse abstract system and return data required to construct model
    mtx = austrip.(hcat(bounding_box(system)...))
    T = eltype(mtx)
    lattice = zeros(T, 3, 3)
    lattice[1:D, 1:D] .= mtx

    # Cache for instantiated pseudopotentials (such that the respective objects are
    # indistinguishable in memory. We need that property to fill potential_groups in Model)
    cached_pseudos = Dict{String,Any}()
    atoms = map(system) do atom
        if hasproperty(atom, :potential)
            potential = atom.potential
        elseif hasproperty(atom, :pseudopotential)
            pspkey = atom.pseudopotential
            if !(pspkey in keys(cached_pseudos))
                cached_pseudos[pspkey] = ElementPsp(atomic_symbol(atom); psp=load_psp(pspkey))
            end
            potential = cached_pseudos[pspkey]
        else
            potential = ElementCoulomb(atomic_symbol(atom))
        end

        coordinate = zeros(T, 3)
        coordinate[1:D] = lattice[1:D, 1:D] \ T.(austrip.(position(atom)))
        potential => Vec3{T}(coordinate)
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

    (; lattice, atoms, kwargs=(; magnetic_moments))
end
