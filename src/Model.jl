# Contains the physical specification of the model

# spin_polarisation values:
#     :none       No spin polarisation, αα and ββ density identical, αβ and βα blocks zero
#     :collinear  Spin is polarised, but on all atoms in the same direction. αα ̸= ββ, αβ = βα = 0
#     :full       Generic magnetisation, any direction on any atom. αβ, βα, αα, ββ all nonzero, different
#     :spinless   No spin at all ("spinless fermions", "mathematicians' electrons").
#                 Difference with :none is that the occupations are 1 instead of 2

struct Model{T <: Real}
    # Lattice and reciprocal lattice vectors in columns
    lattice::Mat3{T}
    recip_lattice::Mat3{T}
    unit_cell_volume::T
    recip_cell_volume::T
    dim::Int # Dimension of the system; 3 unless `lattice` has zero columns

    # Electrons, occupation and smearing function
    n_electrons::Int
    spin_polarisation::Symbol  # :none, :collinear, :full, :spinless
    temperature::T
    smearing
    assume_band_gap::Bool  # DFTK may assume a band gap at the Fermi level to be present

    # Potential definitions and builders
    build_external  # External potential, e.g. local pseudopotential term
    build_nonlocal  # Non-local potential, e.g. non-local pseudopotential projectors
    build_hartree
    build_xc
end


"""
TODO docme

If no smearing is specified the system will be assumed to be an insulator
Occupation obtained as `f(ε) = smearing((ε-εF) / T)`
"""
function Model(lattice::AbstractMatrix{T}, n_electrons; external=nothing,
               nonlocal=nothing, hartree=nothing, xc=nothing, temperature=0.0,
               smearing=nothing, spin_polarisation=:none, assume_band_gap=nothing) where {T <: Real}
    lattice = SMatrix{3, 3, T, 9}(lattice)

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
    unit_cell_volume = det(lattice[1:d, 1:d])
    recip_cell_volume = det(recip_lattice[1:d, 1:d])

    @assert spin_polarisation in (:none, :collinear, :full, :spinless)

    # Default to Fermi-Dirac smearing
    if temperature > 0.0 && smearing === nothing
        smearing = smearing_fermi_dirac
    end

    # Assume a band gap (insulator, semiconductor) if no smearing given
    assume_band_gap === nothing && (assume_band_gap = smearing === nothing)

    build_nothing(args...; kwargs...) = (nothing, nothing)
    Model{T}(lattice, recip_lattice, unit_cell_volume, recip_cell_volume, d, n_electrons,
             spin_polarisation, T(temperature), smearing, assume_band_gap,
             something(external, build_nothing), something(nonlocal, build_nothing),
             something(hartree, build_nothing), something(xc, build_nothing))
end


"""
How many electrons to put in each state.
"""
function filled_occupation(model)
    @assert model.spin_polarisation in (:none, :spinless)
    if model.spin_polarisation == :none
        @assert model.n_electrons % 2 == 0
        filled_occ = 2
    else
        filled_occ = 1
    end
    filled_occ
end
