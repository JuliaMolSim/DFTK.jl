# Data structures for representing the kinetic operator in a particular basis
# and functionality to represent a k-Point block of it

"""
Kinetic energy operator for a particular PlaneWaveBasis basis
"""

# Kinetic energy operator for a particular PlaneWaveBasis basis
# and the kbuild function to build a k-Point-specific block of it.
struct Kinetic
    basis::PlaneWaveBasis
end

function kblock(kin::Kinetic, kpt::Kpoint)
    basis = kin.basis
    model = basis.model
    basis.model.spin_polarisation in (:none, :collinear, :spinless) || (
        error("$(pw.model.spin_polarisation) not implemented"))
    # TODO For spin_polarisation == :full we need to double
    #      the vector (2 full spin components)

    T = eltype(basis.kpoints[1].coordinate)
    Diagonal(Vector{T}([sum(abs2, model.recip_lattice * (G + kpt.coordinate))
                        for G in G_vectors(kpt)] ./ 2))
end
