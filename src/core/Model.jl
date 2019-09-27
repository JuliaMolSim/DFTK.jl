include("types.jl")

# Contains the physical specification of the model

# spin_polarisation values:
#     :none       No spin polarisation, αα and ββ density identical, αβ and βα blocks zero
#     :collinear  Spin is polarised, but on all atoms in the same direction. αα ̸= ββ, αβ = βα = 0
#     :full       Generic magnetisation, any direction on any atom. αβ, βα, αα, ββ all nonzero, different


struct Model{T <: Real}
    # Lattice and reciprocal lattice vectors in columns
    lattice::Mat3{T}
    recip_lattice::Mat3{T}
    unit_cell_volume::T
    recip_cell_volume::T

    # Electrons, occupation and smearing function
    n_electrons::Int
    spin_polarisation::Symbol  # :none, :collinear, :full
    temperature::T
    smearing

    # Potential definitions and builders
    pot_local
    pot_nonlocal
    pot_hartree
    pot_xc
end


"""
TODO docme
"""
function Model(lattice::AbstractMatrix{T}, n_electrons;
               pot_local=nothing, pot_nonlocal=nothing, pot_hartree=nothing, pot_xc=nothing,
               temperature=0.0, smearing=nothing, spin_polarisation=:none) where {T <: Real}
    lattice = SMatrix{3, 3, T, 9}(lattice)
    recip_lattice = 2π * inv(lattice')

    @assert spin_polarisation in [:none, :collinear, :full]
    Model{T}(lattice, recip_lattice, det(lattice), det(recip_lattice), n_electrons,
             spin_polarisation, T(temperature), smearing, pot_local, pot_nonlocal,
             pot_hartree, pot_xc)
end
