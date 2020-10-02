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
             temperature=basis.model.temperature, spins=1:basis.model.n_spin_components)
    N = zero(ε)
    if (temperature == 0) || smearing isa Smearing.None
        error("NOS only supports finite temperature")
    end
    @assert basis.model.spin_polarization in (:none, :spinless, :collinear)

    # Note the differences to the DOS and LDOS functions: We are not counting states
    # per BZ volume (like in DOS), but absolute number of states. Therefore n_symeqk
    # is used instead of kweigths. We count the states inside a temperature window
    # `temperature` centred about εik. For this the states are weighted by the distribution
    # -f'((εik - ε)/temperature).
    #
    # To explicitly show the similarity with DOS and the temperature dependence we employ
    # -f'((εik - ε)/temperature) = temperature * ( d/dε f_τ(εik - ε') )|_{ε' = ε}
    for σ in spins, ik = krange_spin(basis, σ)
        n_symeqk = length(basis.ksymops[ik])  # Number of symmetry-equivalent k-Points
        for (iband, εnk) in enumerate(eigenvalues[ik])
            enred = (εnk - ε) / temperature
            N -= n_symeqk * Smearing.occupation_derivative(smearing, enred)
        end
    end
    N
end


"""
Total density of states at energy ε
"""
function DOS(ε, basis, eigenvalues; smearing=basis.model.smearing,
             temperature=basis.model.temperature, spins=1:basis.model.n_spin_components)
    if (temperature == 0) || smearing isa Smearing.None
        error("DOS only supports finite temperature")
    end
    @assert basis.model.spin_polarization in (:none, :spinless, :collinear)
    filled_occ = filled_occupation(basis.model)

    D = zero(ε)
    for σ in spins, ik = krange_spin(basis, σ)
        for (iband, εnk) in enumerate(eigenvalues[ik])
            enred = (εnk - ε) / temperature
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
              temperature=basis.model.temperature, spins=1:basis.model.n_spin_components)
    if (temperature == 0) || smearing isa Smearing.None
        error("LDOS only supports finite temperature")
    end
    @assert basis.model.spin_polarization in (:none, :spinless, :collinear)
    filled_occ = filled_occupation(basis.model)

    weights = deepcopy(eigenvalues)
    for (ik, εk) in enumerate(eigenvalues)
        for (iband, εnk) in enumerate(εk)
            enred = (εnk - ε) / temperature
            weights[ik][iband] = (-filled_occ / temperature
                                  * Smearing.occupation_derivative(smearing, enred))
        end
    end

    # Use compute_density routine to compute LDOS, using just the modified
    # weights (as "occupations") at each kpoint. Note, that this automatically puts in the
    # required symmetrization with respect to kpoints and BZ symmetry
    ldostot, ldosspin = compute_density(basis, ψ, weights)

    # TODO This is not great, make compute_density more flexible ...
    if basis.model.spin_polarization == :collinear
        ρs = [(ldostot.real + ldosspin.real) / 2, (ldostot.real - ldosspin.real) / 2]
    else
        @assert isnothing(ldosspin)
        ρs = [ldostot.real]
    end
    return sum(ρs[iσ] for iσ in spins)
end
