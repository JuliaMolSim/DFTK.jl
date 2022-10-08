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
    # Useful for conversions between cartesian and reduced coordinates
    inv_lattice::Mat3{T}
    inv_recip_lattice::Mat3{T}
    # Volumes
    unit_cell_volume::T
    recip_cell_volume::T

    # If not `nothing` then do computations in the canonical ensemble, where
    # `n_electrons` is fixed; electron count usually consistent with `atoms` field.
    n_electrons::Union{Int, Nothing}
    # Expert option: If not `nothing` then do computations in the grand-canonical ensemble,
    # where `εF` is fixed. Computations with Coulomb electrostatics will fail by default.
    # Use `check_electrostatics=false` to disable these checks.
    εF::Union{T, Nothing}
    check_electrostatics::Bool  # Enables checks that electrostatics is consistent

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
    Model(lattice, atoms, positions; n_electrons, magnetic_moments, terms, temperature,
          smearing, spin_polarization, symmetries)

Creates the physical specification of a model (without any discretization information).

`n_electrons` is taken from `atoms` if not specified.

`spin_polarization` is :none by default (paired electrons)
unless any of the elements has a non-zero initial magnetic moment.
In this case the spin_polarization will be :collinear.

`magnetic_moments` is only used to determine the symmetry and the
`spin_polarization`; it is not stored inside the datastructure.

`smearing` is Fermi-Dirac if `temperature` is non-zero, none otherwise

The `symmetries` kwarg allows (a) to pass `true` / `false` to enable / disable
the automatic determination of lattice symmetries or (b) to pass an explicit list
of symmetry operations to use for lowering the computational effort.
The default behaviour is equal to `true`, namely that the code checks the
specified model in form of the Hamiltonian `terms`, `lattice`, `atoms` and `magnetic_moments`
parameters and from these automatically determines a set of symmetries it can safely use.
If you want to pass custom symmetry operations (e.g. a reduced or extended set) use the
`symmetry_operations` function. Notice that this may lead to wrong results if e.g. the
external potential breaks some of the passed symmetries. Use `false` to turn off
symmetries completely.
"""
function Model(lattice::AbstractMatrix{T},
               atoms::Vector{<:Element}=Element[],
               positions::Vector{<:AbstractVector}=Vec3{T}[];
               model_name="custom",
               n_electrons::Union{Int, Nothing}=nothing,
               εF=nothing,
               check_electrostatics=any(!iszero, charge_ionic.(atoms)),
               magnetic_moments=T[],
               terms=[Kinetic()],
               temperature=zero(T),
               smearing=default_smearing(temperature),
               spin_polarization=default_spin_polarization(magnetic_moments),
               symmetries=default_symmetries(lattice, atoms, positions, magnetic_moments,
                                             spin_polarization, terms),
               ) where {T <: Real}
    # Ensembles and electrons
    if isnothing(n_electrons) && isnothing(εF)  # Default: NVT with electrons given by atoms
        n_electrons = default_n_electrons(atoms)
    end

    is_μVT = !isnothing(εF)           # Grand-canonical ensemble
    is_NVT = !isnothing(n_electrons)  # Canonical ensemble (default)
    is_μVT && is_NVT && error("`n_electrons` is incompatible with fixed Fermi level `εF`.")
    something(n_electrons, 0) < 0 && error("n_electrons should be non-negative.")

    if check_electrostatics
        if is_μVT && any(!iszero, charge_ionic.(atoms))
            error("DFTK is currently unable to do Coulomb electrostratics in the " *
                  "grand-canonical ensemble. Don't use any charged atoms.")
        end
        if is_NVT && sum(charge_ionic, atoms; init=0) != n_electrons
            error("DFTK is currently unable to consistently simulate non-neutral cells.")
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
                           n_electrons, εF, check_electrostatics,
                           spin_polarization, n_spin, temperature, smearing,
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

normalize_magnetic_moment(::Nothing)::Vec3{Float64}          = (0, 0, 0)
normalize_magnetic_moment(mm::Number)::Vec3{Float64}         = (0, 0, mm)
normalize_magnetic_moment(mm::AbstractVector)::Vec3{Float64} = mm


"""Defaults to Fermi-Dirac smearing when finite temperature."""
default_smearing(temperature) = temperature > 0 ? Smearing.FermiDirac() : Smearing.None()

"""Defaults to the number of valence electrons."""
default_n_electrons(atoms)    = sum(n_elec_valence, atoms; init=0)

"""
`:none` if no element has a magnetic moment, else `:collinear` or `:full`.
"""
function default_spin_polarization(magnetic_moments)
    isempty(magnetic_moments) && return :none
    all_magmoms = normalize_magnetic_moment.(magnetic_moments)
    all(iszero, all_magmoms) && return :none
    all(iszero(magmom[1:2]) for magmom in all_magmoms) && return :collinear

    :full
end

"""
Default logic to determine the symmetry operations to be used in the model.
"""
function default_symmetries(lattice, atoms, positions, magnetic_moments, spin_polarization,
                            terms; tol_symmetry=SYMMETRY_TOLERANCE)
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

# Ensembles
is_NVT(model::Model) = !isnothing(model.n_electrons)
is_μVT(model::Model) = !isnothing(model.εF)

function assert_consistent_electrostatics(model::Model)
    # DFTK currently assumes in a number of terms that the compensating charge
    # in the electronic and nuclear terms is equal and of opposite sign.
    # See also the PSP correction term, where n_electrons is used synonymously
    # for sum of charges.
    if model.check_electrostatics
        @assert !is_μVT
        @assert sum(charge_ionic, model.atoms) == model.n_electrons
    end
end


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
=#
vector_red_to_cart(model::Model, rred)        = model.lattice * rred
vector_cart_to_red(model::Model, rcart)       = model.inv_lattice * rcart
covector_red_to_cart(model::Model, fred)      = model.inv_lattice' * fred
covector_cart_to_red(model::Model, fcart)     = model.lattice' * fcart
recip_vector_red_to_cart(model::Model, qred)  = model.recip_lattice * qred
recip_vector_cart_to_red(model::Model, qcart) = model.inv_recip_lattice * qcart

#=
Transformations on vectors and covectors are matrices and comatrices.

Consider two covectors f and g related by a transformation matrix B. In reduced
coordinates g_red = B_red f_red and in cartesian coordinates we want g_cart = B_cart f_cart.
From g_cart = L⁻ᵀ g_red = L⁻ᵀ B_red f_red = L⁻ᵀ B_red Lᵀ f_cart, we see B_cart = L⁻ᵀ B_red Lᵀ.

Similarly for two vectors r and s with s_red = A_red r_red and s_cart = A_cart r_cart:
s_cart = L s_red = L A_red r_red = L A_red L⁻¹ r_cart, thus A_cart = L A_red L⁻¹.

Examples of matrices are the symmetries in real space (W)
Examples of comatrices are the symmetries in reciprocal space (S)
=#
matrix_red_to_cart(model::Model, Ared)    = model.lattice * Ared * model.inv_lattice
matrix_cart_to_red(model::Model, Acart)   = model.inv_lattice * Acart * model.lattice
comatrix_red_to_cart(model::Model, Bred)  = model.inv_lattice' * Bred * model.lattice'
comatrix_cart_to_red(model::Model, Bcart) = model.lattice' * Bcart * model.inv_lattice'
