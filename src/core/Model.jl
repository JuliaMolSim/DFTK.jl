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
    recip_lattice = 2π * inv(lattice')
    @assert spin_polarisation in (:none, :collinear, :full, :spinless)

    # Assume a band gap (insulator, semiconductor) if no smearing given
    assume_band_gap === nothing && (assume_band_gap = smearing === nothing)

    # Default to Fermi-Dirac smearing
    temperature > 0.0 && smearing === nothing && (smearing = smearing_fermi_dirac)

    build_nothing(args...; kwargs...) = (nothing, nothing)
    Model{T}(lattice, recip_lattice, det(lattice), det(recip_lattice), n_electrons,
             spin_polarisation, T(temperature), smearing, assume_band_gap,
             something(external, build_nothing), something(nonlocal, build_nothing),
             something(hartree, build_nothing), something(xc, build_nothing))
end
