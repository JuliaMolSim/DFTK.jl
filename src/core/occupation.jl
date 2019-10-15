import Roots

# Functions for finding the Fermi level and occupation numbers for bands
"""
Find Fermi level and occupation for the given parameters, assuming a band gap
and zero temperature.
"""
function find_occupation_gap_zero_temperature(basis, energies, Psi)
    n_bands = size(Psi[1], 2)
    n_electrons = basis.model.n_electrons
    T = eltype(basis.kpoints[1].coordinate)

    @assert basis.model.spin_polarisation == :none
    @assert n_electrons % 2 == 0
    n_occ = div(n_electrons, 2)
    @assert n_bands ≥ n_occ
    @assert basis.model.temperature == 0.0

    HOMO = -Inf
    LUMO = Inf
    occupation = similar(basis.kpoints, Vector{T})
    for ik in 1:length(occupation)
        occupation[ik] = zeros(T, n_bands)
        occupation[ik][1:n_occ] .= 2
        HOMO = max(HOMO, energies[ik][n_occ])
        if n_occ < n_bands
            LUMO = min(LUMO, energies[ik][n_occ + 1])
        end
    end
    @assert sum(basis.kweights .* sum.(occupation)) ≈ n_electrons

    # Put Fermi level slightly above HOMO energy
    εF = HOMO + 10eps(eltype(energies[1]))
    if HOMO > LUMO || εF > LUMO
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
    model = basis.model
    n_bands = size(Psi[1], 2)
    n_electrons = model.n_electrons
    temperature = model.temperature
    smearing = model.smearing

    @assert smearing !== nothing
    @assert basis.model.spin_polarisation == :none
    @assert n_electrons % 2 == 0
    @assert n_bands ≥ n_electrons / 2
    @assert sum(basis.kweights) ≈ 1

    # Avoid some numerical issues
    temperature == 0 && (smearing(x) = x ≤ 0 ? 1 : 0)

    compute_occupation(εF) = [2 * smearing.((ε .- εF) ./ temperature) for ε in energies]
    compute_n_elec(εF) = sum(basis.kweights .* sum.(compute_occupation(εF)))

    min_ε = minimum([minimum(ε) for ε in energies]) - 1
    max_ε = maximum([maximum(ε) for ε in energies]) + 1
    @assert compute_n_elec(min_ε) < n_electrons < compute_n_elec(max_ε)

    # Just use bisection here; note that with MP smearing there might
    # be multiple possible Fermi levels. This could be sped up with more
    # advanced methods (e.g. false position), but more care has to be
    # taken with convergence criteria and the like
    εF = Roots.find_zero(εF -> compute_n_elec(εF) - n_electrons, (min_ε, max_ε),
                         Roots.Bisection())
    @assert compute_n_elec(εF) ≈ n_electrons
    εF, compute_occupation(εF)
end


"""
Find the Fermi level, given a `basis`, SCF band `energies` and corresponding Bloch
one-particle wave function `Psi`. If `basis.model.assume_band_gap` this function
assumes a band gap at the Fermi level.
"""
function find_fermi_level(basis, energies, Psi)
    εF = nothing
    if basis.model.assume_band_gap && model.temperature == 0.0
        εF, _ = compute_occupation_gap_zero_temperature(basis, energies, Psi)
    elseif basis.model.assume_band_gap && model.temperature > 0.0
        error("`model.assume_band_gap` and `model.temperature > 0.0` not implemented.")
    else
        εF, _ = find_fermi_level_metal(basis, energies, Psi)
    end
    εF
end

"""
Find the occupation numbers around the Fermi level, given a `basis`, SCF band `energies`
and corresponding Bloch one-particle wave function `Psi`. If `basis.model.smearing` is
`nothing`, this function assumes an insulator.
"""
function find_occupation_around_fermi(basis, energies, Psi)
    model = basis.model
    occ = nothing
    if model.assume_band_gap && basis.model.temperature == 0.0
        _, occ = find_occupation_gap_zero_temperature(basis, energies, Psi)
    elseif model.assume_band_gap && model.temperature > 0.0
        error("`model.assume_band_gap` and `model.temperature > 0.0` not implemented.")
    else
        _, occ = find_occupation_fermi_metal(basis, energies, Psi)
    end
    occ
end
