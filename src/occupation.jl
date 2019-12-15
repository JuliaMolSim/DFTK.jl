# Functions for finding the Fermi level and occupation numbers for bands

import Roots

"""
Find Fermi level and occupation for the given parameters, assuming a band gap
and zero temperature.
"""
function find_occupation_gap_zero_temperature(basis, energies, Psi)
    @assert basis.model.spin_polarisation in (:none, :spinless)
    n_bands = size(Psi[1], 2)
    n_electrons = basis.model.n_electrons
    T = eltype(basis.kpoints[1].coordinate)
    @assert basis.model.temperature == 0.0

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
        @warn("`find_occupation_gap_zero_temperature` assumes an insulator, but the " *
              "system seems metallic. Try specifying a temperature and a smearing function.",
              HOMO, LUMO, maxlog=5)
    end

    εF, occupation
end

"""
Find the Fermi level and occupation for the given parameters, assuming no band gap
(i.e. a metal).
"""
function find_occupation_fermi_metal(basis, energies, Psi)
    @assert basis.model.spin_polarisation in (:none, :spinless)
    model = basis.model
    n_bands = size(Psi[1], 2)
    n_electrons = model.n_electrons
    temperature = model.temperature
    smearing = model.smearing

    @assert smearing !== nothing

    filled_occ = filled_occupation(model)

    @assert filled_occ*n_bands ≥ n_electrons

    # Avoid numerical issues by imposing a step function for zero temp
    temperature == 0 && (smearing(x) = x ≤ 0 ? 1 : 0)

    # Find εF so that
    # n_i = filled_occ * f((εi-εF)/T)
    # sum_i n_i = n_electrons
    compute_occupation(εF) = [filled_occ * smearing.((ε .- εF) ./ temperature) for ε in energies]
    compute_n_elec(εF) = sum(basis.kweights .* sum.(compute_occupation(εF)))

    # Get rough bounds to bracket εF
    min_ε = minimum(minimum.(energies)) - 1
    max_ε = maximum(maximum.(energies)) + 1
    @assert compute_n_elec(min_ε) < n_electrons < compute_n_elec(max_ε)

    # Just use bisection here; note that with MP smearing there might
    # be multiple possible Fermi levels. This could be sped up with more
    # advanced methods (e.g. false position), but more care has to be
    # taken with convergence criteria and the like
    εF = Roots.find_zero(εF -> compute_n_elec(εF) - n_electrons, (min_ε, max_ε),
                         Roots.Bisection())
    @assert compute_n_elec(εF) ≈ n_electrons
    minocc = maximum(minimum.(compute_occupation(εF)))
    if minocc > .01
        @warn "One kpoint has a high minimum occupation $minocc. You should probably increase the number of bands."
    end
    εF, compute_occupation(εF)
end


"""
Find the Fermi level and occupation numbers, given a `basis`, SCF band `energies`
and corresponding Bloch one-particle wave function `Psi`. If `basis.model.assume_band_gap`
this function assumes a band gap at the Fermi level.
Returns `(fermi_level, occupation)`
"""
function find_occupation(basis, energies, Psi)
    occ = nothing
    if basis.model.assume_band_gap && basis.model.temperature == 0.0
        εF, occ = find_occupation_gap_zero_temperature(basis, energies, Psi)
    elseif basis.model.assume_band_gap && basis.model.temperature > 0.0
        error("`model.assume_band_gap` and `model.temperature > 0.0` not implemented.")
    else
        εF, occ = find_occupation_fermi_metal(basis, energies, Psi)
    end
    εF, occ
end

"""
Find the Fermi level, given a `basis`, SCF band `energies` and corresponding Bloch
one-particle wave function `Psi`. If `basis.model.assume_band_gap` this function
assumes a band gap at the Fermi level.
"""
function find_fermi_level(basis, energies, Psi)
    εF, _ = find_occupation(basis, energies, Psi)
    εF
end
