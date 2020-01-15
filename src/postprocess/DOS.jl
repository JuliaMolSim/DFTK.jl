# Compute densities of states

# IDOS (integrated density of states)
# N(ε) = sum_n f_n = sum_n f((εn-ε)/T)
# DOS (density of states)
# D(ε) = N'(ε)
# LDOS (local density of states)
# LD = sum_n f_n ψn

using ForwardDiff

@doc raw"""
    NOS(ε, basis, orben; smearing=basis.model.smearing, T=basis.model.temperature)

The number of Kohn-Sham states in a temperature window of width T around the energy ε
contributing to the DOS at temperature T.

This quantity is not a physical quantity, much rather a dimensionaless approximate measure
for how well properties near the Fermi surface are sampled with the passed `smearing`
and temperature `T`. It increases with both `T` and better sampling of the BZ with
``k``-Points. A value ``\gg 1`` indicates a good sampling of properties near the
Fermi surface.
"""
function NOS(ε, basis, orben; smearing=basis.model.smearing, T=basis.model.temperature)
    N = zero(ε)
    T != 0 || error("NOS only supports finite temperature")

    # Note the differences to the DOS and LDOS functions: We are not counting states
    # per BZ volume (like in DOS), but absolute number of states. Therefore n_symeqk
    # is used instead of kweigths. The number of states is counted over a temperature
    # window of width T, where we assume that the number of states remains constant
    # in the interval [T - T/2, T + T/2] as ( d/dε f_T(εik - ε') )|_{ε' = ε}
    # with f_T the smearing function. This gives the factor T in front.
    for ik = 1:length(orben)
        n_symeqk = length(basis.ksymops[ik])  # Number of symmetry-equivalent k-Points
        for iband = 1:length(orben[ik])
            N += (T * n_symeqk
                    * ForwardDiff.derivative(ε -> smearing((orben[ik][iband] - ε) / T), ε))
        end
    end
    N
end


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
