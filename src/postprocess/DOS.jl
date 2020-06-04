# Compute densities of states

# IDOS (integrated density of states)
# N(ε) = sum_n f_n = sum_n f((εn-ε)/temperature)
# DOS (density of states)
# D(ε) = N'(ε)
# LDOS (local density of states)
# LD = sum_n f_n ψn

using ForwardDiff

@doc raw"""
    NOS(ε, basis, eigenvalues; smearing=basis.model.smearing,
        temperature=basis.model.temperature)

The number of Kohn-Sham states in a temperature window of width `temperature` around the
energy `ε` contributing to the DOS at temperature `T`.

This quantity is not a physical quantity, but rather a dimensionless approximate measure
for how well properties near the Fermi surface are sampled with the passed `smearing`
and temperature `T`. It increases with both `T` and better sampling of the BZ with
``k``-Points. A value ``\gg 1`` indicates a good sampling of properties near the
Fermi surface.
"""
function NOS(ε, basis, eigenvalues; smearing=basis.model.smearing,
             temperature=basis.model.temperature)
    N = zero(ε)
    if (temperature == 0) || smearing isa Smearing.None
        error("NOS only supports finite temperature")
    end

    # Note the differences to the DOS and LDOS functions: We are not counting states
    # per BZ volume (like in DOS), but absolute number of states. Therefore n_symeqk
    # is used instead of kweigths. We count the states inside a temperature window
    # `temperature` centred about εik. For this the states are weighted by the distribution
    # -f'((εik - ε)/temperature).
    #
    # To explicitly show the similarity with DOS and the temperature dependence we employ
    # -f'((εik - ε)/temperature) = temperature * ( d/dε f_τ(εik - ε') )|_{ε' = ε}
    for ik = 1:length(eigenvalues)
        n_symeqk = length(basis.ksymops[ik])  # Number of symmetry-equivalent k-Points
        for iband = 1:length(eigenvalues[ik])
            enred = (eigenvalues[ik][iband] - ε) / temperature
            N -= n_symeqk * Smearing.occupation_derivative(smearing, enred)
        end
    end
    N
end


"""
Total density of states at energy ε
"""
function DOS(ε, basis, eigenvalues; smearing=basis.model.smearing,
             temperature=basis.model.temperature)
    filled_occ = filled_occupation(basis.model)
    D = zero(ε)
    if (temperature == 0) || smearing isa Smearing.None
        error("DOS only supports finite temperature")
    end
    for ik = 1:length(eigenvalues)
        for iband = 1:length(eigenvalues[ik])
            enred = (eigenvalues[ik][iband] - ε) / temperature
            D -= (filled_occ * basis.kweights[ik] / temperature
                  * Smearing.occupation_derivative(smearing, enred))
        end
    end
    D
end

"""
Local density of states, in real space
"""
function LDOS(ε, basis, eigenvalues, ψ; smearing=basis.model.smearing,
              temperature=basis.model.temperature)
    filled_occ = filled_occupation(basis.model)
    if (temperature == 0) || smearing isa Smearing.None
        error("LDOS only supports finite temperature")
    end
    weights = deepcopy(eigenvalues)
    for ik = 1:length(eigenvalues)
        for iband = 1:length(eigenvalues[ik])
            enred = (eigenvalues[ik][iband] - ε) / temperature
            weights[ik][iband] = (-filled_occ / temperature
                                  * Smearing.occupation_derivative(smearing, enred))
        end
    end

    # Use compute_density routine to compute LDOS, using just the modified
    # weights (as "occupations") at each kpoint. Note, that this automatically puts in the
    # required symmetrization with respect to kpoints and BZ symmetry
    compute_density(basis, ψ, weights).real
end
