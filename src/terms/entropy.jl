"""
Entropy term -TS, where S is the electronic entropy.
Turns the energy E into the free energy F=E-TS.
This is in particular useful because the free energy,
not the energy, is minimized at self-consistency.
"""
struct Entropy end
(::Entropy)(basis) = TermEntropy(basis)

struct TermEntropy <: Term
    basis::PlaneWaveBasis
end

function ene_ops(term::TermEntropy, ψ, occ; kwargs...)
    basis = term.basis
    T = eltype(basis)
    ops = [NoopOperator(term.basis, kpoint) for kpoint in basis.kpoints]

    smearing = basis.model.smearing
    temperature = basis.model.temperature

    temperature == 0 && return (E=zero(T), ops=ops)
    ψ == nothing && return (E=T(Inf), ops=ops)

    filled_occ = filled_occupation(basis.model)
    eigenvalues = kwargs[:eigenvalues]
    εF = kwargs[:εF]

    E = zero(T)
    for (ik, k) in enumerate(basis.kpoints)
        for iband = 1:size(ψ[1], 2)
            E -= (temperature
                  * basis.kweights[ik]
                  * filled_occ
                  * Smearing.entropy(smearing, (eigenvalues[ik][iband] - εF) / temperature))
        end
    end
    E = MPI.Allreduce(E, +, basis.mpi_kcomm)

    (E=E, ops=ops)
end
