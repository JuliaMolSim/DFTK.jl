"""
Entropy term -TS, where S is the electronic entropy.
Turns the energy E into the free energy F=E-TS.
This is in particular useful because the free energy,
not the energy, is minimized at self-consistency.
"""
struct Entropy end
(::Entropy)(basis) = TermEntropy()
struct TermEntropy <: Term end

function ene_ops(term::TermEntropy, basis::PlaneWaveBasis{T}, ψ, occ; kwargs...) where {T}
    ops = [NoopOperator(basis, kpt) for kpt in basis.kpoints]
    smearing    = basis.model.smearing
    temperature = basis.model.temperature

    iszero(temperature) && return (E=zero(T), ops=ops)
    isnothing(ψ)        && return (E=T(Inf),  ops=ops)

    !(:εF in keys(kwargs))          && return (E=T(Inf), ops=ops)
    !(:eigenvalues in keys(kwargs)) && return (E=T(Inf), ops=ops)
    εF = kwargs[:εF]
    eigenvalues = kwargs[:eigenvalues]

    E = zero(T)
    for (ik, k) in enumerate(basis.kpoints)
        for iband = 1:size(ψ[1], 2)
            E -= (temperature
                  * basis.kweights[ik]
                  * filled_occupation(basis.model)
                  * Smearing.entropy(smearing, (eigenvalues[ik][iband] - εF) / temperature))
        end
    end
    E = mpi_sum(E, basis.comm_kpts)

    (E=E, ops=ops)
end
