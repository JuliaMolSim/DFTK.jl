# Contains the physical specification of the model

# A physical specification of a model.
# Contains the geometry information, but no discretization parameters.
# The exact model used is defined by the list of terms.
struct Model{T <: Real, VT <: Real}
    # T is the default type to express data, VT the corresponding bare value type (i.e. not dual)

    # Human-readable name for the model (like LDA, PBE, ...)
    model_name::String

    # Lattice and reciprocal lattice vectors in columns
    lattice::Mat3{T}
    recip_lattice::Mat3{T}
    # Dimension of the system; 3 unless `lattice` has zero columns
    n_dim::Int
    # Useful for conversions between Cartesian and reduced coordinates
    inv_lattice::Mat3{T}
    inv_recip_lattice::Mat3{T}
    # Volumes
    unit_cell_volume::T
    recip_cell_volume::T

    # Computations can be performed at fixed `n_electrons` (`n_electrons` Int, `εF` nothing),
    # or fixed Fermi level (expert option, `n_electrons` nothing, `εF` T)
    n_electrons::Union{Int, Nothing}
    εF::Union{T, Nothing}

    # spin_polarization values:
    #     :none       No spin polarization, αα and ββ density identical,
    #                 αβ and βα blocks zero, 1 spin component treated explicitly
    #     :collinear  Spin is polarized, but everywhere in the same direction.
    #                 αα ̸= ββ, αβ = βα = 0, 2 spin components treated
    #     :full       Generic magnetization, non-uniform direction.
    #                 αβ, βα, αα, ββ all nonzero, different (not supported)
    #     :spinless   No spin at all ("spinless fermions", "mathematicians' electrons").
    #                 The difference with :none is that the occupations are 1 instead of 2
    spin_polarization::Symbol
    n_spin_components::Int  # 2 if :collinear, 1 otherwise

    # If temperature==0, no fractional occupations are used.
    # If temperature is nonzero, the occupations are
    # `fn = max_occ*smearing((εn-εF) / temperature)`
    temperature::T
    smearing::Smearing.SmearingFunction # see Smearing.jl for choices

    # Particle types (elements) and particle positions and in fractional coordinates.
    # Possibly empty. It's up to the `term_types` to make use of this (or not).
    # `atom_groups` contains the groups of indices into atoms and positions, which
    # point to identical atoms. It is computed automatically on Model construction and may
    # be used to optimise the term instantiation.
    atoms::Vector{Element}
    positions::Vector{Vec3{T}}  # positions[i] is the location of atoms[i] in fract. coords
    atom_groups::Vector{Vector{Int}}  # atoms[i] == atoms[j] for all i, j in atom_group[α]

    # each element t must implement t(basis), which instantiates a
    # term in a given basis and gives back a term (<: Term)
    # see terms.jl for some default terms
    term_types::Vector

    # list of symmetries of the model
    symmetries::Vector{SymOp{VT}}
end

_is_well_conditioned(A; tol=1e5) = (cond(A) <= tol)

"""
    Model(lattice, atoms, positions; model_name, εF, n_electrons,
                disable_electrostatics_check, magnetic_moments, terms,
                temperature, smearing, spin_polarization, symmetries)

Creates the physical specification of a model (without any discretization information).
See also [`Model(system::AbstractSystem; kwargs...)`](@ref) for an AtomsBase-compatible
constructor.

The `Model` represents the model system through its geometry (lattice, atomic positions),
electronic structure (number of electrons, spin polarization), and physical terms
(kinetic energy, external potentials, Hartree, exchange-correlation, etc.). 
It does not include discretization parameters; those are set separately when constructing
a [`PlaneWaveBasis`](@ref) from a `Model`.

## Arguments
- `lattice::AbstractMatrix`: Lattice vectors given column-wise (atomic units by default).
    Alternatively, a matrix of [`Unitful.Quantity`](@ref) can be provided,
    which is then converted to atomic units automatically.
- `atoms::Vector{<:Element}`: Atomic species (may include pseudopotential metadata).
- `positions::Vector{Vec3}`: Atomic positions in fractional (reduced) coordinates.

## Keyword arguments
- `n_electrons::Int` (default: inferred from `atoms`): Total electron count.
- `magnetic_moments` (default: `[]`): Per-atom initial magnetic moments used to infer
    `spin_polarization` and to determine symmetries. These values influence model setup
    but are otherwise not stored on the `Model`.
- `terms::Vector` (default: `[Kinetic()]`): List of energy term types to include in the
    Hamiltonian.
- `temperature::Real` (default: `0`): Electronic temperature.
    If `> 0`, fractional occupations are enabled and `smearing` defaults to
    [`Smearing.FermiDirac`](@ref)().
- `smearing::Smearing.SmearingFunction` (default: `FermiDirac()` when `temperature>0`,
    else `None()`):
    Smearing function used to compute occupations from Kohn-Sham eigenvalues.
- `spin_polarization::Symbol` (default: `determine_spin_polarization(magnetic_moments)`):
    Controls spin treatment; allowed values are `:none`, `:collinear`, `:spinless`.
- `symmetries` (default: `true`):
    - `true`: run automatic symmetry detection with [`default_symmetries`](@ref).
    - `false`: disable all symmetries.
    - `::Vector{SymOp}`: use the provided explicit symmetry operations. Passing incorrect
        symmetries may produce wrong results.

## Keyword arguments (Expert level)
- `model_name::String` (default: "custom"): Human-readable model name (e.g. "LDA", "PBE").
- `εF` (default: `nothing`): Fixed Fermi level option. If set, the model is defined at fixed
    Fermi level instead of fixed electron count. Mutually exclusive with `n_electrons`.

## Notes
- `magnetic_moments` are used only to determine `spin_polarization` and symmetries,
    but are otherwise not stored on the `Model`.
- The constructor validates consistency between `εF` and `n_electrons` and will error if
    both are supplied.
- Lattice conditioning is checked and a warning is issued for poorly conditioned lattices
    (high matrix condition number).

## See also
- [`Model` datastructure](@ref)
- [`Model(system::AbstractSystem; kwargs...)`](@ref)
- [`PlaneWaveBasis`](@ref), [`self_consistent_field`](@ref)
"""
function Model(lattice::AbstractMatrix{Tstatic},
               atoms::Vector{<:Element}=Element[],
               positions::Vector{<:AbstractVector}=Vec3{Tstatic}[];
               model_name="custom",
               εF=nothing,
               n_electrons::Union{Int,Nothing}=isnothing(εF) ?
                                               n_electrons_from_atoms(atoms) : nothing,
               # Force electrostatics with non-neutral cells; results not guaranteed.
               # Set to `true` by default for charged systems.
               disable_electrostatics_check=all(iszero, charge_ionic.(atoms)),
               magnetic_moments=Tstatic[],
               terms=[Kinetic()],
               temperature=zero(Tstatic),
               smearing=temperature > 0 ? Smearing.FermiDirac() : Smearing.None(),
               spin_polarization=determine_spin_polarization(magnetic_moments),
               symmetries=default_symmetries(lattice, atoms, positions, magnetic_moments,
                                             spin_polarization, terms),
               ) where {Tstatic <: Real}
    # # a bit convoluted because kwargs can't determine type parameters
    T = promote_type(Tstatic, typeof(temperature))

    # Validate εF and n_electrons
    if !isnothing(εF)  # fixed Fermi level
        if !isnothing(n_electrons)
            error("Cannot have both a given `n_electrons` and a fixed Fermi level `εF`.")
        end
        if !disable_electrostatics_check
            error("Coulomb electrostatics is incompatible with fixed Fermi level.")
        end
    else  # fixed number of electrons
        n_electrons < 0 && error("n_electrons should be non-negative.")
        if !disable_electrostatics_check && n_electrons_from_atoms(atoms) != n_electrons
            error("Support for non-neutral cells is experimental and likely broken.")
        end
    end

    # Atoms and terms
    if length(atoms) != length(positions)
        error("Length of atoms and positions vectors need to agree.")
    end
    isempty(terms) && error("Model without terms not supported.")
    atom_groups = [findall(Ref(pot) .== atoms) for pot in Set(atoms)]

    # Special handling of 1D and 2D systems, and sanity checks
    lattice = Mat3{T}(lattice)
    n_dim = count(!iszero, eachcol(lattice))
    n_dim > 0 || error("Check your lattice; we do not do 0D systems")
    for i = n_dim+1:3
        norm(lattice[:, i]) == norm(lattice[i, :]) == 0 || error(
            "For 1D and 2D systems, the non-empty dimensions must come first")
    end
    _is_well_conditioned(lattice[1:n_dim, 1:n_dim]) || @warn (
        "Your lattice is badly conditioned, the computation is likely to fail.")

    # Note: In the 1D or 2D case, the volume is the length/surface
    inv_lattice       = compute_inverse_lattice(lattice)
    recip_lattice     = compute_recip_lattice(lattice)
    inv_recip_lattice = compute_inverse_lattice(recip_lattice)
    unit_cell_volume  = compute_unit_cell_volume(lattice)
    recip_cell_volume = compute_unit_cell_volume(recip_lattice)

    # Spin polarization
    spin_polarization in (:none, :collinear, :full, :spinless) ||
        error("Only :none, :collinear, :full and :spinless allowed for spin_polarization")
    spin_polarization == :full && error("Full spin polarization not yet supported")
    !isempty(magnetic_moments) && !(spin_polarization in (:collinear, :full)) && @warn(
        "Non-empty magnetic_moments on a Model without spin polarization detected."
    )
    n_spin = length(spin_components(spin_polarization))

    temperature = T(austrip(temperature))
    temperature < 0 && error("temperature must be non-negative")

    if !allunique(string.(nameof.(typeof.(terms))))
        error("Having several terms of the same name is not supported.")
    end

    # Determine symmetry operations to use
    if symmetries === true
        symmetries = default_symmetries(lattice, atoms, positions, magnetic_moments,
                                        spin_polarization, terms)
    elseif symmetries === false
        symmetries = [one(SymOp)]
    end
    @assert !isempty(symmetries)  # Identity has to be always present.

    Model{T,value_type(T)}(model_name,
                           lattice, recip_lattice, n_dim, inv_lattice, inv_recip_lattice,
                           unit_cell_volume, recip_cell_volume,
                           n_electrons, εF, spin_polarization, n_spin, temperature, smearing,
                           atoms, positions, atom_groups, terms, symmetries)
end
function Model(lattice::AbstractMatrix{<:Integer}, atoms::Vector{<:Element},
               positions::Vector{<:AbstractVector}; kwargs...)
    Model(Float64.(lattice), atoms, positions; kwargs...)
end
function Model(lattice::AbstractMatrix{<:Quantity}, atoms::Vector{<:Element},
               positions::Vector{<:AbstractVector}; kwargs...)
    Model(austrip.(lattice), atoms, positions; kwargs...)
end


"""
    Model(system::AbstractSystem; kwargs...)

AtomsBase-compatible Model constructor. Sets structural information (`atoms`, `positions`,
`lattice`, `n_electrons` etc.) from the passed `system`.

## Keyword arguments
- `pseudopotentials`: Set the pseudopotential information for the atoms
   of the passed system. Can be (a) a list of pseudopotential objects
   (one for each atom), where a `nothing` element indicates that the
   Coulomb potential should be used for that atom or (b)
   a `PseudoPotentialData.PseudoFamily` to automatically determine the
   pseudopotential from the specified pseudo family or (c)
   a `Dict{Symbol,String}` mapping an atomic symbol
   to the pseudopotential to be employed.
- `kwargs...`: Additional keyword arguments passed to the main
   [`Model(::AbstractMatrix, ::Vector, ::Vector)`](@ref) constructor.
"""
function Model(system::AbstractSystem;
               pseudopotentials=fill(nothing, length(system)), kwargs...)
    parsed = parse_system(system, pseudopotentials)
    Model(parsed.lattice, parsed.atoms, parsed.positions;
          parsed.magnetic_moments, kwargs...)
end

"""
    Model(model; [lattice, positions, atoms, kwargs...])
    Model{T}(model; [lattice, positions, atoms, kwargs...])

Construct an identical model to `model` with the option to change some of
the contained parameters. This constructor is useful for changing the data
type in the model or for changing `lattice` or `positions` in
geometry/lattice optimisations.

!!! warning "Symmetries need to be updated manually."
    Unless a `symmetries` keyword argument is given
    this constructor copies the symmetries from `model`.
    This means if you update the `positions`, `lattice`, `atoms`
    or `terms` in a way that they *break* some of the original symmetries,
    you should either also supply matching `symmetries` or specify
    `symmetries=true`, which will re-determine the symmetries corresponding
    to the new structure. In the case of spin do also re-supply the
    appropriate `magnetic_moments` of the atoms.

    This behaviour is slated to change in an upcoming breaking release.
"""
function Model{T}(model::Model;
                  lattice::AbstractMatrix=model.lattice,
                  positions::Vector{<:AbstractVector}=model.positions,
                  atoms::Vector{<:Element}=model.atoms,
                  symmetries=model.symmetries,
                  magnetic_moments=T[],
                  kwargs...) where {T <: Real}
    # TODO In an upcoming breaking release come up with a way to store the symmetry-equivalent
    #      atoms in the model (including the effect magnetic moments), such that if the atoms
    #      do not change (the usual case) the default symmetry determination routine can still
    #      work with updated positions without the need to re-provide the magnetic_moments.

    Model(T.(lattice), atoms, positions;
          model.model_name,
          model.n_electrons,
          magnetic_moments=T.(magnetic_moments),
          terms=model.term_types,
          temperature=T(model.temperature),
          model.smearing,
          model.εF,
          model.spin_polarization,
          symmetries,
          # Can be safely disabled: this has been checked for model
          disable_electrostatics_check=true,
          kwargs...
    )
end
function Model(model::Model{T};
               lattice::AbstractMatrix{U}=model.lattice,
               positions::Vector{<:AbstractVector}=model.positions,
               kwargs...) where {T, U}
    TT = promote_type(T, U, eltype(positions[1]))
    Model{TT}(model; lattice, positions, kwargs...)
end

Base.convert(::Type{Model{T}}, model::Model{T}) where {T}    = model
Base.convert(::Type{Model{U}}, model::Model{T}) where {T, U} = Model{U}(model)

normalize_magnetic_moment(::Nothing)::Vec3{Float64}          = (0, 0, 0)
normalize_magnetic_moment(mm::Number)::Vec3{Float64}         = (0, 0, mm)
normalize_magnetic_moment(mm::AbstractVector)::Vec3{Float64} = mm

"""Number of valence electrons."""
n_electrons_from_atoms(atoms) = sum(n_elec_valence, atoms; init=0)

"""
Default logic to determine the symmetry operations to be used in the model.
"""
function default_symmetries(lattice, atoms, positions, magnetic_moments,
                            spin_polarization, terms;
                            tol_symmetry=SYMMETRY_TOLERANCE)
    dimension = count(!iszero, eachcol(lattice))
    if spin_polarization == :full || dimension != 3
        return [one(SymOp)]  # Symmetry not supported in spglib
    elseif spin_polarization == :collinear && isempty(magnetic_moments)
        # Spin-breaking due to initial magnetic moments cannot be determined
        return [one(SymOp)]
    elseif any(breaks_symmetries, terms)
        return [one(SymOp)]  # Terms break symmetry
    end

    # Standard case from here on:
    if length(positions) != length(atoms)
        error("Length of atoms and positions vectors need to agree.")
    end
    if !isempty(magnetic_moments) && length(magnetic_moments) != length(atoms)
        error("Length of atoms and magnetic_moments vectors need to agree.")
    end
    magnetic_moments = normalize_magnetic_moment.(magnetic_moments)
    symmetry_operations(lattice, atoms, positions, magnetic_moments; tol_symmetry)
end


"""
Maximal occupation of a state (2 for non-spin-polarized electrons, 1 otherwise).
"""
function filled_occupation(model)
    if model.spin_polarization in (:spinless, :collinear)
        return 1
    elseif model.spin_polarization == :none
        return 2
    else
        error("Not implemented $(model.spin_polarization)")
    end
end


"""
Explicit spin components of the KS orbitals and the density
"""
function spin_components(spin_polarization::Symbol)
    spin_polarization == :collinear && return (:up, :down  )
    spin_polarization == :none      && return (:both,      )
    spin_polarization == :spinless  && return (:spinless,  )
    spin_polarization == :full      && return (:undefined, )
end
spin_components(model::Model) = spin_components(model.spin_polarization)

# prevent broadcast
import Base.Broadcast.broadcastable
Base.Broadcast.broadcastable(model::Model) = Ref(model)

#=
There are two types of quantities, depending on how they transform under change of coordinates.

Positions transform with the lattice: r_cart = lattice * r_red. We term them vectors.

Linear forms on vectors (anything that appears in an expression f⋅r) transform
with the inverse lattice transpose: if f_cart ⋅ r_cart = f_red ⋅ r_red, then
f_cart = lattice' \ f_red. We term them covectors.
Examples of covectors are forces.

Reciprocal vectors are a special case: they are covectors, but conventionally have an
additional factor of 2π in their definition, so they transform rather with 2π times the
inverse lattice transpose: q_cart = 2π lattice' \ q_red = recip_lattice * q_red.

For each of the function there is a one-argument version (returning a function to do the
transformation) and a two-argument version applying the transformation to a passed vector.
=#
@inline _gen_matmul(mat) = vec -> mat * vec

vector_red_to_cart(model::Model)       = _gen_matmul(model.lattice)
vector_cart_to_red(model::Model)       = _gen_matmul(model.inv_lattice)
covector_red_to_cart(model::Model)     = _gen_matmul(model.inv_lattice')
covector_cart_to_red(model::Model)     = _gen_matmul(model.lattice')
recip_vector_red_to_cart(model::Model) = _gen_matmul(model.recip_lattice)
recip_vector_cart_to_red(model::Model) = _gen_matmul(model.inv_recip_lattice)

vector_red_to_cart(model::Model, vec)       = vector_red_to_cart(model)(vec)
vector_cart_to_red(model::Model, vec)       = vector_cart_to_red(model)(vec)
covector_red_to_cart(model::Model, vec)     = covector_red_to_cart(model)(vec)
covector_cart_to_red(model::Model, vec)     = covector_cart_to_red(model)(vec)
recip_vector_red_to_cart(model::Model, vec) = recip_vector_red_to_cart(model)(vec)
recip_vector_cart_to_red(model::Model, vec) = recip_vector_cart_to_red(model)(vec)

#=
Transformations on vectors and covectors are matrices and comatrices, i.e. matrices act on
a vector to give a vector and comatrices act on covector to give a covector.
Beware that some quantities (e.g., the force constant matrix) map vectors to covectors and
are therefore not matrices nor comatrices; they fall outside of these methods.

Consider two covectors f and g related by a transformation matrix B. In reduced
coordinates g_red = B_red f_red and in Cartesian coordinates we want g_cart = B_cart f_cart.
From g_cart = L⁻ᵀ g_red = L⁻ᵀ B_red f_red = L⁻ᵀ B_red Lᵀ f_cart, we see B_cart = L⁻ᵀ B_red Lᵀ.

Similarly for two vectors r and s with s_red = A_red r_red and s_cart = A_cart r_cart:
s_cart = L s_red = L A_red r_red = L A_red L⁻¹ r_cart, thus A_cart = L A_red L⁻¹.

Examples of matrices are the symmetries in real space (W)
Examples of comatrices are the symmetries in reciprocal space (S)
=#
@inline _gen_matmatmul(M, Minv) = mat -> M * mat * Minv

matrix_red_to_cart(model::Model)   = _gen_matmatmul(model.lattice,      model.inv_lattice)
matrix_cart_to_red(model::Model)   = _gen_matmatmul(model.inv_lattice,  model.lattice)
comatrix_red_to_cart(model::Model) = _gen_matmatmul(model.inv_lattice', model.lattice')
comatrix_cart_to_red(model::Model) = _gen_matmatmul(model.lattice',     model.inv_lattice')

matrix_red_to_cart(model::Model, Ared)    = matrix_red_to_cart(model)(Ared)
matrix_cart_to_red(model::Model, Acart)   = matrix_cart_to_red(model)(Acart)
comatrix_red_to_cart(model::Model, Bred)  = comatrix_red_to_cart(model)(Bred)
comatrix_cart_to_red(model::Model, Bcart) = comatrix_cart_to_red(model)(Bcart)
