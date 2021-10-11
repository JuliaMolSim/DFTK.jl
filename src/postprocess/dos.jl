# Compute densities of states

# IDOS (integrated density of states)
# N(ε) = sum_n f_n = sum_n f((εn-ε)/temperature)
# DOS (density of states)
# D(ε) = N'(ε)
# LDOS (local density of states)
# LD = sum_n f_n ψn

@doc raw"""
    compute_nos(ε, basis, eigenvalues; smearing=basis.model.smearing,
                temperature=basis.model.temperature)

The number of Kohn-Sham states in a temperature window of width `temperature` around the
energy `ε` contributing to the DOS at temperature `T`.

This quantity is not a physical quantity, but rather a dimensionless approximate measure
for how well properties near the Fermi surface are sampled with the passed `smearing`
and temperature `T`. It increases with both `T` and better sampling of the BZ with
``k``-points. A value ``\gg 1`` indicates a good sampling of properties near the
Fermi surface.
"""
function compute_nos(ε, basis, eigenvalues; smearing=basis.model.smearing,
                     temperature=basis.model.temperature)
    N = zeros(typeof(ε), basis.model.n_spin_components)
    if (temperature == 0) || smearing isa Smearing.None
        error("compute_nos only supports finite temperature")
    end

    # Note the differences to the DOS and LDOS functions: We are not counting states
    # per BZ volume (like in DOS), but absolute number of states. Therefore n_symeqk
    # is used instead of kweigths. We count the states inside a temperature window
    # `temperature` centred about εik. For this the states are weighted by the distribution
    # -f'((εik - ε)/temperature).
    #
    # To explicitly show the similarity with DOS and the temperature dependence we employ
    # -f'((εik - ε)/temperature) = temperature * ( d/dε f_τ(εik - ε') )|_{ε' = ε}
    for σ in 1:basis.model.n_spin_components, ik = krange_spin(basis, σ)
        n_symeqk = length(basis.ksymops[ik])  # Number of symmetry-equivalent k-points
        for (iband, εnk) in enumerate(eigenvalues[ik])
            enred = (εnk - ε) / temperature
            N[σ] -= n_symeqk * Smearing.occupation_derivative(smearing, enred)
        end
    end
    N = mpi_sum(N, basis.comm_kpts)
end


"""
Total density of states at energy ε
"""
function compute_dos(ε, basis, eigenvalues; smearing=basis.model.smearing,
                     temperature=basis.model.temperature)
    if (temperature == 0) || smearing isa Smearing.None
        error("compute_dos only supports finite temperature")
    end
    filled_occ = filled_occupation(basis.model)

    D = zeros(typeof(ε), basis.model.n_spin_components)
    for σ in 1:basis.model.n_spin_components, ik = krange_spin(basis, σ)
        for (iband, εnk) in enumerate(eigenvalues[ik])
            enred = (εnk - ε) / temperature
            D[σ] -= (filled_occ * basis.kweights[ik] / temperature
                     * Smearing.occupation_derivative(smearing, enred))
        end
    end
    D = mpi_sum(D, basis.comm_kpts)
end

"""
Local density of states, in real space
"""
function compute_ldos(ε, basis, eigenvalues, ψ; smearing=basis.model.smearing,
                      temperature=basis.model.temperature)
    if (temperature == 0) || smearing isa Smearing.None
        error("compute_ldos only supports finite temperature")
    end
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
    # weights (as "occupations") at each k-point. Note, that this automatically puts in the
    # required symmetrization with respect to kpoints and BZ symmetry
    compute_density(basis, ψ, weights)
end

"""
Plot the density of states over a reasonable range
"""
function plot_dos end
