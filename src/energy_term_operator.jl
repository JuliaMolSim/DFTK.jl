# Helper functions to compute individual energy terms

"""
Compute the energy term resulting from an operator `op`, a Bloch wavefunction `Psi`
and the passed `occupation` vector.
"""
function energy_term_operator(op, Psi, occupation)
    basis = op.basis
    @assert length(occupation) == length(Psi)
    @assert length(occupation) == length(basis.kpoints)
    @assert length(occupation) == length(basis.kweights)
    real(sum(basis.kweights[ik]
             * sum(occupation[ik] .*
                   [dot(psi, kblock(op, kpt) * psi) for psi in eachcol(Psi[ik])])
             for (ik, kpt) in enumerate(basis.kpoints)
    ))
end
