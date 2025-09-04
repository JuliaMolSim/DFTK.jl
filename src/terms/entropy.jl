"""
Entropy term ``-TS``, where ``S`` is the electronic entropy.
Turns the energy ``E`` into the free energy ``F=E-TS``.
This is in particular useful because the free energy,
not the energy, is minimized at self-consistency.
"""
function compute_entropy(basis::PlaneWaveBasis{T}, ψ, εF, eigenvalues) where {T}
    smearing    = basis.model.smearing
    temperature = basis.model.temperature
    E = zero(T)
    for ik in 1:length(basis.kpoints)
        for iband = 1:size(ψ[ik], 2)
            E -= (temperature
                  * basis.kweights[ik]
                  * filled_occupation(basis.model)
                  * Smearing.entropy(smearing, (eigenvalues[ik][iband] - εF) / temperature))
        end
    end
    mpi_sum(E, basis.comm_kpts)
end
