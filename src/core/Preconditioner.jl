# Data structures for representing Preconditioners for selected
# hamiltonians and their k-Point blocks

struct PreconditionerKinetic
    basis::PlaneWaveModel
    α
end

"""Kinetic-energy based preconditioner.

Applies ``1 / (|k + G|^2 / 2 + α)`` to the vectors, when called with `ldiv!`.
This attempts to dampen the high-kinetic energy parts of the
Hamiltonian, thus making the Hamiltonian more well-conditioned.
"""
function PreconditionerKinetic(ham::Hamiltonian; α=0)
    PreconditionerKinetic(ham.basis, α)
end


function kblock(prec::PreconditionerKinetic, kpt::Kpoint)
    basis = prec.basis
    model = basis.model
    basis.model.spin_polarisation in [:none, :collinear] || (
        error("$(pw.model.spin_polarisation) not implemented"))
    # TODO For spin_polarisation == :full we need to double
    #      the vector (2 full spin components)

    T = eltype(basis.kpoints[1].coordinate)
    qsq = Vector{T}([sum(abs2, model.recip_lattice * (G + kpt.coordinate))
                     for G in kpt.basis] ./ 2)
    Diagonal(qsq .+ prec.α)
end
