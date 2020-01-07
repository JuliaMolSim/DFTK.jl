# Compute densities of states

# IDOS (integrated density of states)
# N(ε) = sum_n f_n = sum_n f((εn-ε)/T)
# DOS (density of states)
# D(ε) = N'(ε)
# LDOS (local density of states)
# LD = sum_n f_n ψn

using ForwardDiff

"""
Total density of states at energy ε
"""
function DOS(ε, basis, orben; smearing=basis.model.smearing, T=basis.model.temperature)
    filled_occ = filled_occupation(basis.model)
    D = zero(ε)
    T != 0 || error("DOS only supports finite temperature")
    for ik = 1:length(orben)
        for iband = 1:length(orben[ik])
            D += (filled_occ * basis.kweights[ik]
                  * ForwardDiff.derivative(ε -> smearing((orben[ik][iband] - ε) / T), ε)
                 )
        end
    end
    D
end

"""
Local density of states, in real space
"""
function LDOS(ε, basis, orben, Psi; smearing=basis.model.smearing, T=basis.model.temperature)
    filled_occ = filled_occupation(basis.model)
    D = zeros(real(eltype(Psi[1])), basis.fft_size)
    T != 0 || error("LDOS only supports finite temperature")
    for ik = 1:length(orben)
        ψreal = G_to_r(basis, basis.kpoints[ik], Psi[ik])
        for iband = 1:length(orben[ik])
            D += (filled_occ * basis.kweights[ik]
                  * ForwardDiff.derivative(ε -> smearing((orben[ik][iband] - ε) / T), ε)
                  * abs2.(ψreal[:, :, :, iband]))
        end
    end
    D
end
