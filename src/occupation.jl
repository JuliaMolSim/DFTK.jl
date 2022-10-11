# Functions for finding the Fermi level and occupation numbers for bands

import Roots


"""
Find the occupation and Fermi level.
"""
function compute_occupation(basis::PlaneWaveBasis, eigenvalues;
                            temperature=basis.model.temperature)
    if !isnothing(model.εF) # fixed Fermi level
        εF = basis.model.εF
    else # fixed n_electrons
        εF = compute_fermi_level(basis, eigenvalues; temperature)
    end
    occupation = compute_occupation(basis, eigenvalues, εF; temperature)

    (; occupation, εF)
end


"""Compute the occupations, given eigenenergies and a Fermi level"""
function compute_occupation(basis::PlaneWaveBasis{T}, eigenvalues, εF;
                            temperature=basis.model.temperature,
                            smearing=basis.model.smearing) where {T}
    # This is needed to get the right behaviour for special floating-point types
    # such as intervals.
    inverse_temperature = iszero(temperature) ? T(Inf) : 1/temperature

    filled_occ = filled_occupation(basis.model)
    occupation = [filled_occ * Smearing.occupation.(smearing, (εk .- εF) .* inverse_temperature)
                  for εk in eigenvalues]
    occupation
end

function compute_fermi_level(basis::PlaneWaveBasis{T}, eigenvalues; temperature) where {T}
    n_electrons = basis.model.n_electrons
    n_spin      = basis.model.n_spin_components

    # Maximum occupation per state
    filled_occ = filled_occupation(basis.model)

    # The goal is to find εF so that
    # n_i = filled_occ * f((εi-εF)/θ) with θ = temperature
    # sum_i n_i = n_electrons
    # If temperature is zero, (εi-εF)/θ = ±∞.
    # The occupation function is required to give 1 and 0 respectively in these cases.
    function compute_n_elec(εF)
        weighted_ksum(basis, sum.(compute_occupation(basis, eigenvalues, εF; temperature)))
    end

    if filled_occ * weighted_ksum(basis, length.(eigenvalues)) < (n_electrons - sqrt(eps(T)))
        error("Could not obtain required number of electrons by filling every state. " *
              "Increase n_bands.")
    end

    # Get rough bounds to bracket εF
    min_ε = minimum(minimum, eigenvalues) - 1
    min_ε = mpi_min(min_ε, basis.comm_kpts)
    max_ε = maximum(maximum, eigenvalues) + 1
    max_ε = mpi_max(max_ε, basis.comm_kpts)

    if iszero(temperature)
        # Sanity check that we can indeed fill the appropriate number of states
        if n_electrons % (n_spin * filled_occ) != 0
            error("$n_electrons electrons cannot be attained by filling states with " *
                  "occupation $filled_occ. Typically this indicates that you need to put " *
                  "a temperature or switch to a calculation with collinear spin polarization.")
        end

        # For zero temperature, two cases arise: either there are as many bands
        # as electrons, in which case we set εF to the highest energy level
        # reached, or there are unoccupied conduction bands and we take
        # εF as the midpoint between valence and conduction bands.
        if compute_n_elec(max_ε) ≈ n_electrons
            εF = max_ε
        else
            # The sanity check above ensures that n_fill is well defined
            n_fill = div(n_electrons, n_spin * filled_occ, RoundUp)
            # highest occupied energy level
            HOMO = maximum([εk[n_fill] for εk in eigenvalues])
            HOMO = mpi_max(HOMO, basis.comm_kpts)
            # lowest unoccupied energy level, be careful that not all k-points
            # might have at least n_fill+1 energy levels so we have to take care
            # of that by specifying init to minimum
            LUMO = minimum(minimum.([εk[n_fill+1:end] for εk in eigenvalues]; init=T(Inf)))
            LUMO = mpi_min(LUMO, basis.comm_kpts)
            εF = (HOMO + LUMO) / 2
        end
    else
        # For finite temperature, just use bisection; note that with MP smearing
        # there might be multiple possible Fermi levels. This could be sped up
        # with more advanced methods (e.g. false position), but more care has to
        # be taken with convergence criteria and the like
        @assert compute_n_elec(min_ε) < n_electrons < compute_n_elec(max_ε)
        εF = Roots.find_zero(εF -> compute_n_elec(εF) - n_electrons, (min_ε, max_ε),
                             Roots.Bisection(), atol=eps(T))
    end

    if !isapprox(compute_n_elec(εF), n_electrons)
        if iszero(temperature)
            error("Unable to find non-fractional occupations that have the " *
                  "correct number of electrons. You should add a temperature.")
        else
            error("This should not happen, debug me!")
        end
    end
    εF
end
