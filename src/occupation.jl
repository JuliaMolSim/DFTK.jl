# Functions for finding the Fermi level and occupation numbers for bands

import Roots

"""
Find the Fermi level.
"""
function find_fermi_level(basis, energies)
    find_occupation(basis, energies)[1]
end

"""
Find the occupation and Fermi level.
"""
function find_occupation(basis::PlaneWaveBasis{T}, energies) where {T}
    @assert basis.model.spin_polarisation in (:none, :spinless)
    model = basis.model
    n_electrons = model.n_electrons
    temperature = model.temperature
    smearing = model.smearing

    # Avoid numerical issues by imposing a step function for zero temp
    temperature == 0 && (smearing(x) = x ≤ 0 ? 1 : 0)
    @assert smearing !== nothing

    # Maximum occupation per state
    filled_occ = filled_occupation(model)

    # The goal is to find εF so that
    # n_i = filled_occ * f((εi-εF)/T)
    # sum_i n_i = n_electrons
    compute_occupation(εF) = [filled_occ * smearing.((ε .- εF) ./ temperature) for ε in energies]
    compute_n_elec(εF) = sum(basis.kweights .* sum.(compute_occupation(εF)))

    # assume that we can get the required number of electrons by filling every state
    @assert filled_occ*sum(basis.kweights .* length.(energies)) ≥ n_electrons - sqrt(eps(T))

    # Get rough bounds to bracket εF
    min_ε = minimum(minimum.(energies)) - 1
    max_ε = maximum(maximum.(energies)) + 1
    if temperature != 0
        @assert compute_n_elec(min_ε) < n_electrons < compute_n_elec(max_ε)
    end

    # Compute εF by bisection
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
                             Roots.Bisection())
    end

    if !isapprox(compute_n_elec(εF), n_electrons)
        if temperature == 0
            error("Unable to find non-fractional occupations that have the" *
                  "correct number of electrons. You should add a temperature.")
        else
            error("This should not happen, debug me!")
        end
    end

    minocc = maximum(minimum.(compute_occupation(εF)))
    if minocc > .01
        @warn "One kpoint has a high minimum occupation $minocc. You should probably increase the number of bands."
    end

    (occupation=compute_occupation(εF), εF=εF)
end

"""
Find Fermi level and occupation for the given parameters, assuming a band gap
and zero temperature. This function is for DEBUG purposes only, and the
finite-temperature version with 0 temperature should be preferred.
"""
function find_occupation_bandgap(basis, energies)
    @assert basis.model.spin_polarisation in (:none, :spinless)
    n_bands = length(energies[1])
    @assert all(e -> length(e) == n_bands, energies)
    n_electrons = basis.model.n_electrons
    T = eltype(basis.kpoints[1].coordinate)
    @assert basis.model.temperature == 0

    filled_occ = filled_occupation(basis.model)
    n_fill = div(n_electrons, filled_occ)
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
    @assert sum(basis.kweights .* sum.(occupation)) ≈ n_electrons

    # Put Fermi level slightly above HOMO energy, to ensure that HOMO < εF
    εF = nextfloat(HOMO)
    if εF > LUMO
        @warn("`find_occupation_zero_temperature` assumes an insulator, but the " *
              "system seems metallic. Try specifying a temperature and a smearing function.",
              HOMO, LUMO, maxlog=5)
    end

    (occupation=occupation, εF=εF)
end
