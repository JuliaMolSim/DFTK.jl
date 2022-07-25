# Functions for finding the Fermi level and occupation numbers for bands

import Roots

"""
Find the Fermi level.
"""
fermi_level(basis, energies) = compute_occupation(basis, energies).εF

"""
Find the occupation and Fermi level.
"""
function compute_occupation(basis::PlaneWaveBasis{T}, energies;
                            temperature=basis.model.temperature,
                            smearing=basis.model.smearing) where {T}
    n_electrons = basis.model.n_electrons

    # Maximum occupation per state
    filled_occ = filled_occupation(basis.model)

    if temperature == 0 && n_electrons % filled_occ != 0
        error("$n_electrons electron cannot be attained by filling states with " *
              "occupation $filled_occ. Typically this indicates that you need to put " *
              "a temperature or switch to a calculation with collinear spin polarization.")
    end

    # The goal is to find εF so that
    # n_i = filled_occ * f((εi-εF)/T)
    # sum_i n_i = n_electrons
    # If temperature is zero, (εi-εF)/T = ±∞.
    # The occupation function is required to give 1 and 0 respectively in these cases.
    compute_occupation(εF) = [filled_occ * Smearing.occupation.(smearing, (ε .- εF) ./ temperature)
                              for ε in energies]
    compute_n_elec(εF) = weighted_ksum(basis, sum.(compute_occupation(εF)))

    if filled_occ * weighted_ksum(basis, length.(energies)) < (n_electrons - sqrt(eps(T)))
        error("Could not obtain required number of electrons by filling every state. " *
              "Increase n_bands.")
    end

    # Get rough bounds to bracket εF
    min_ε = minimum(minimum.(energies)) - 1
    min_ε = mpi_min(min_ε, basis.comm_kpts)
    max_ε = maximum(maximum.(energies)) + 1
    max_ε = mpi_max(max_ε, basis.comm_kpts)
    if temperature != 0
        @assert compute_n_elec(min_ε) < n_electrons < compute_n_elec(max_ε)
    end

    if compute_n_elec(max_ε) ≈ n_electrons
        # This branch takes care of the case of insulators at zero
        # temperature with as many bands as electrons; there it is
        # possible that compute_n_elec(max_ε) ≈ n_electrons but
        # compute_n_elec(max_ε) < n_electrons, so that bisection does
        # not work
        εF = max_ε
    else
        # Just use bisection here; note that with MP smearing there might
        # be multiple possible Fermi levels. This could be sped up with more
        # advanced methods (e.g. false position), but more care has to be
        # taken with convergence criteria and the like
        εF = Roots.find_zero(εF -> compute_n_elec(εF) - n_electrons, (min_ε, max_ε),
                             Roots.Bisection(), atol=eps(T))
    end

    if !isapprox(compute_n_elec(εF), n_electrons)
        # For insulators it can happen that bisection stops in a final interval (a, b) where
        # `compute_n_elec(a) ≈ n_electrons` and `compute_n_elec(b) > n_electrons`, but where
        # the returned `(a+b)/2` is rounded to `b`, such that `εF` gives a too
        # large electron count. To make sure this is not the case, make εF a little smaller.
        εF -= eps(εF)
    end

    if !isapprox(compute_n_elec(εF), n_electrons)
        if temperature == 0
            error("Unable to find non-fractional occupations that have the " *
                  "correct number of electrons. You should add a temperature.")
        else
            error("This should not happen, debug me!")
        end
    end

    minocc = maximum(minimum.(compute_occupation(εF)))
    if temperature > 0 && minocc > 1e-5
        @warn "One k-point has a high minimum occupation $minocc. You should probably increase the number of bands."
    end

    (occupation=compute_occupation(εF), εF=εF)
end

"""
Find Fermi level and occupation for the given parameters, assuming a band gap
and zero temperature. This function is for DEBUG purposes only, and the
finite-temperature version with 0 temperature should be preferred.
"""
function compute_occupation_bandgap(basis, energies)
    n_bands = length(energies[1])
    @assert all(e -> length(e) == n_bands, energies)
    n_electrons = basis.model.n_electrons
    T = eltype(basis)
    @assert basis.model.temperature == 0

    filled_occ = filled_occupation(basis.model)
    n_fill = div(n_electrons, filled_occ, RoundUp)
    @assert filled_occ * n_fill == n_electrons
    @assert n_bands ≥ n_fill

    # We need to fill n_fill states with occupation filled_occ
    # Find HOMO and LUMO
    HOMO = -Inf # highest occupied energy state
    LUMO = Inf  # lowest unoccupied energy state
    occupation = similar(basis.kpoints, Vector{T})
    for ik in 1:length(occupation)
        occupation[ik] = zeros(T, n_bands)
        occupation[ik][1:n_fill] .= filled_occ
        HOMO = max(HOMO, energies[ik][n_fill])
        if n_fill < n_bands
            LUMO = min(LUMO, energies[ik][n_fill + 1])
        end
    end
    LUMO = mpi_min(LUMO, basis.comm_kpts)
    HOMO = mpi_max(HOMO, basis.comm_kpts)
    @assert weighted_ksum(basis, sum.(occupation)) ≈ n_electrons

    # Put Fermi level slightly above HOMO energy, to ensure that HOMO < εF
    εF = nextfloat(HOMO)
    if εF > LUMO
        @warn("`compute_occupation_bandgap` assumes an insulator, but the " *
              "system seems metallic. Try specifying a temperature and a smearing function.",
              HOMO, LUMO)
    end

    (occupation=occupation, εF=εF)
end
