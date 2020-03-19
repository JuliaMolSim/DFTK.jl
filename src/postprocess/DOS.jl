# Compute densities of states

# IDOS (integrated density of states)
# N(ε) = sum_n f_n = sum_n f((εn-ε)/T)
# DOS (density of states)
# D(ε) = N'(ε)
# LDOS (local density of states)
# LD = sum_n f_n ψn

using ForwardDiff

@doc raw"""
    NOS(ε, basis, eigenvalues; smearing=basis.model.smearing, T=basis.model.temperature)

The number of Kohn-Sham states in a temperature window of width T around the energy ε
contributing to the DOS at temperature T.

This quantity is not a physical quantity, but rather a dimensionless approximate measure
for how well properties near the Fermi surface are sampled with the passed `smearing`
and temperature `T`. It increases with both `T` and better sampling of the BZ with
``k``-Points. A value ``\gg 1`` indicates a good sampling of properties near the
Fermi surface.
"""
function NOS(ε, basis, eigenvalues; smearing=basis.model.smearing, T=basis.model.temperature)
    N = zero(ε)
    T != 0 || error("NOS only supports finite temperature")

    # Note the differences to the DOS and LDOS functions: We are not counting states
    # per BZ volume (like in DOS), but absolute number of states. Therefore n_symeqk
    # is used instead of kweigths. We count the states inside a temperature window T
    # centred about εik. For this the states are weighted by the distribution
    # -f'((εik - ε)/T).
    #
    # To explicitly show the similarity with DOS and the T dependence we employ
    # -f'((εik - ε)/T) = T * ( d/dε f_τ(εik - ε') )|_{ε' = ε}
    for ik = 1:length(eigenvalues)
        n_symeqk = length(basis.ksymops[ik])  # Number of symmetry-equivalent k-Points
        for iband = 1:length(eigenvalues[ik])
            N -= (n_symeqk *
                  Smearing.occupation_derivative(smearing, (eigenvalues[ik][iband] - ε) / T))
        end
    end
    N
end


"""
Total density of states at energy ε
"""
function DOS(ε, basis, eigenvalues; smearing=basis.model.smearing, T=basis.model.temperature)
    filled_occ = filled_occupation(basis.model)
    D = zero(ε)
    T != 0 || error("DOS only supports finite temperature")
    for ik = 1:length(eigenvalues)
        for iband = 1:length(eigenvalues[ik])
            D -= (filled_occ * basis.kweights[ik] / T *
                  Smearing.occupation_derivative(smearing, (eigenvalues[ik][iband] - ε) / T))
        end
    end
    D
end

"""
Local density of states, in real space
"""
function LDOS(ε, basis, eigenvalues, ψ; smearing=basis.model.smearing, T=basis.model.temperature)
    filled_occ = filled_occupation(basis.model)
    T != 0 || error("LDOS only supports finite temperature")
    weights = deepcopy(eigenvalues)
    for ik = 1:length(eigenvalues)
        for iband = 1:length(eigenvalues[ik])
            x = (eigenvalues[ik][iband] - ε) / T
            weights[ik][iband] = -filled_occ / T * Smearing.occupation_derivative(smearing, x)
        end
    end

    # Use compute_density routine to compute LDOS, using just the modified
    # weights (as "occupations") at each kpoint. Note, that this automatically puts in the
    # required symmetrisation with respect to kpoints and BZ symmetry
    compute_density(basis, ψ, weights).real
end
