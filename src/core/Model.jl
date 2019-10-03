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
    build_external  # External potential, e.g. local pseudopotential term
    build_nonlocal  # Non-local potential, e.g. non-local pseudopotential projectors
    build_hartree
    build_xc
end


"""
TODO docme
"""
function Model(lattice::AbstractMatrix{T}, n_electrons; external=nothing,
               nonlocal=nothing, hartree=nothing, xc=nothing, temperature=0.0,
               smearing=nothing, spin_polarisation=:none) where {T <: Real}
    lattice = SMatrix{3, 3, T, 9}(lattice)
    recip_lattice = 2π * inv(lattice')
    @assert spin_polarisation in (:none, :collinear, :full)

    build_nothing(args...; kwargs...) = nothing   # Function always returning nothing
    Model{T}(lattice, recip_lattice, det(lattice), det(recip_lattice), n_electrons,
             spin_polarisation, T(temperature), smearing,
             something(ext, build_nothing), something(nonlocal, build_nothing),
             something(hartree, build_nothing), something(xc, build_nothing))
end

# TODO wrapper functions to make "standard models" like DFT with arbitrary xc or so
