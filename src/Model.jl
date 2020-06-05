# Contains the physical specification of the model

# A physical specification of a model.
# Contains the geometry information, but no discretization parameters.
# The exact model used is defined by the list of terms.
struct Model{T <: Real}
    # Lattice and reciprocal lattice vectors in columns
    lattice::Mat3{T}
    recip_lattice::Mat3{T}
    unit_cell_volume::T
    recip_cell_volume::T
    dim::Int # Dimension of the system; 3 unless `lattice` has zero columns

    # Electrons, occupation and smearing function
    n_electrons::Int # usually consistent with `atoms` field, but doesn't have to

    # spin_polarization values:
    #     :none       No spin polarization, αα and ββ density identical,
    #                 αβ and βα blocks zero
    #     :collinear  Spin is polarized, but everywhere in the same direction.
    #                 αα ̸= ββ, αβ = βα = 0
    #     :full       Generic magnetization, non-uniform direction.
    #                 αβ, βα, αα, ββ all nonzero, different
    #     :spinless   No spin at all ("spinless fermions", "mathematicians' electrons").
    #                 The difference with :none is that the occupations are 1 instead of 2
    spin_polarization::Symbol

    # If temperature=0, no fractional occupations are used.
    # If temperature is nonzero, the occupations are
    # `fn = max_occ*smearing((εn-εF) / temperature)`
    temperature::T
    smearing::Smearing.SmearingFunction # see Smearing.jl for choices

    # Vector of pairs Element => vector of vec3 (positions, fractional coordinates)
    # Possibly empty.
    # `atoms` just contain the information on the atoms, it's up to the `terms` to make use of it (or not)
    atoms::Vector{Pair}

    # each element t must implement t(basis), which instantiates a
    # term in a given basis and gives back a term (<: Term)
    # see terms.jl for some default terms
    term_types::Vector

    # list of symmetries of the model
    symops::Vector{SymOp}
end

"""
    Model(lattice; n_electrons, atoms, terms, temperature,
                   smearing, spin_polarization, symmetry)

Creates the physical specification of a model (without any
discretization information).

`n_electrons` is taken from `atoms` if not specified

`spin_polarization` is :none by default (paired electrons)

`smearing` is Fermi-Dirac if `temperature` is non-zero, none otherwise

The `symmetry` kwarg can be:
- auto: determine from terms if they respect the symmetry of the lattice/atoms.
- off: no symmetries at all
- force: force all the symmetries of the lattice/atoms.
Careful that in this last case, wrong results can occur if the
external potential breaks symmetries (this is not checked).
"""
function Model(lattice::AbstractMatrix{T};
               n_electrons=nothing,
               atoms=[],
               terms=[],
               temperature=T(0.0),
               smearing=nothing,
               spin_polarization=:none,
               symmetry=:auto
               ) where {T <: Real}

    lattice = Mat3{T}(lattice)

    if n_electrons === nothing
        # get it from the atom list
        isempty(atoms) && error("Either n_electrons or a non-empty atoms list should be provided")
        n_electrons = sum(length(pos) * n_elec_valence(spec) for (spec, pos) in atoms)
    else
        @assert n_electrons isa Int
    end

    # Special handling of 1D and 2D systems, and sanity checks
    d = 3-count(c -> norm(c) == 0, eachcol(lattice))
    d > 0 || error("Check your lattice; we do not do 0D systems")
    for i = d+1:3
        norm(lattice[:, i]) == norm(lattice[i, :]) == 0 || error(
            "For 1D and 2D systems, the non-empty dimensions must come first")
    end
    cond(lattice[1:d, 1:d]) > 1e-5 || @warn "Your lattice is badly conditioned, the computation is likely to fail"

    # Compute reciprocal lattice and volumes.
    # recall that the reciprocal lattice is the set of G vectors such
    # that G.R ∈ 2π ℤ for all R in the lattice
    recip_lattice = zeros(T, 3, 3)
    recip_lattice[1:d, 1:d] = 2T(π)*inv(lattice[1:d, 1:d]')
    recip_lattice = Mat3{T}(recip_lattice)
    # in the 1D or 2D case, the volume is the length/surface
    unit_cell_volume = abs(det(lattice[1:d, 1:d]))
    recip_cell_volume = abs(det(recip_lattice[1:d, 1:d]))

    @assert spin_polarization in (:none, :collinear, :full, :spinless)

    if smearing === nothing
        @assert temperature >= 0
        # Default to Fermi-Dirac smearing when finite temperature
        smearing = temperature > 0.0 ? Smearing.FermiDirac() : Smearing.None()
    end

    if !allunique(string.(nameof.(typeof.(terms))))
        error("Having several terms of the same name is not supported.")
    end

    @assert symmetry in (:auto, :force, :off)
    if symmetry == :auto
        # check dimension : in 1D or 2D we might have symmetries but it's not
        # detected by spglib so we ignore them
        if d < 3
            compute_symmetry = false
        else
            # ask the terms if they break symmetry
            compute_symmetry = !(any(breaks_symmetries, terms))
        end
    else
        compute_symmetry = (symmetry == :force)
    end

    if compute_symmetry
        symops = symmetry_operations(lattice, atoms)
    else
        symops = [identity_symop()]
    end

    Model{T}(lattice, recip_lattice, unit_cell_volume, recip_cell_volume, d, n_electrons,
             spin_polarization, T(temperature), smearing, atoms, terms, symops)
end

"""
Convenience constructor, which builds a standard atomic (kinetic + atomic potential) model.
Use `extra_terms` to add additional terms.
"""
function model_atomic(lattice::AbstractMatrix, atoms::Vector; extra_terms=[], kwargs...)
    @assert !(:terms in keys(kwargs))
    @assert !(:atoms in keys(kwargs))
    terms = [Kinetic(),
             AtomicLocal(),
             AtomicNonlocal(),
             Ewald(),
             PspCorrection(),
             extra_terms...]
    if :temperature in keys(kwargs) && kwargs[:temperature] != 0
        terms = [terms..., Entropy()]
    end
    Model(lattice; atoms=atoms, terms=terms, kwargs...)
end


"""
Build a DFT model from the specified atoms, with the specified functionals.
"""
function model_DFT(lattice::AbstractMatrix, atoms::Vector, functionals; extra_terms=[], kwargs...)
    model_atomic(lattice, atoms; extra_terms=[Hartree(), Xc(functionals), extra_terms...], kwargs...)
end


"""
Build an LDA model (Teter93 parametrization) from the specified atoms.
"""
function model_LDA(lattice::AbstractMatrix, atoms::Vector; kwargs...)
    model_DFT(lattice, atoms, :lda_xc_teter93; kwargs...)
end


"""
Maximal occupation of a state (2 for non-spin-polarized electrons, 1 otherwise).
"""
function filled_occupation(model)
    @assert model.spin_polarization in (:none, :spinless)
    if model.spin_polarization == :none
        @assert model.n_electrons % 2 == 0
        filled_occ = 2
    else
        filled_occ = 1
    end
    filled_occ
end
