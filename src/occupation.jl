import Roots

# Functions for finding the Fermi level and occupation numbers for bands
# The goal is to find εF so that
# f_i = filled_occ * f((εi-εF)/θ) with θ = temperature
# sum_i f_i = n_electrons
# If temperature is zero, (εi-εF)/θ = ±∞.
# The occupation function is required to give 1 and 0 respectively in these cases.
#
# For finite temperature we right now just use bisection; note that with MP smearing
# there might be multiple possible Fermi levels. This could be sped up
# with more advanced methods (e.g. false position), but more care has to
# be taken with convergence criteria and the like

abstract type FermiLevelAlgorithm end
struct ZeroTemperature <: FermiLevelAlgorithm end
struct Bisection       <: FermiLevelAlgorithm end
struct GaussianNewton  <: FermiLevelAlgorithm end

function deviation_n_electrons(basis::PlaneWaveBasis, eigenvalues, εF; smearing, temperature)
    occupation = compute_occupation(basis, eigenvalues, εF;
                                    smearing, temperature).occupation
    weighted_ksum(basis, sum.(occupation)) - basis.model.n_electrons
end


"""
Find the occupation and Fermi level.
"""
function compute_occupation(basis::PlaneWaveBasis{T}, eigenvalues,
                            algorithm::FermiLevelAlgorithm=GaussianNewton();
                            temperature=basis.model.temperature,
                            smearing=basis.model.smearing) where {T}
    if !isnothing(basis.model.εF)  # fixed Fermi level
        εF = basis.model.εF
    else  # fixed n_electrons
        # Check there are enough bands
        max_n_electrons = (filled_occupation(basis.model)
                           * weighted_ksum(basis, length.(eigenvalues)))
        if max_n_electrons < n_electrons - sqrt(eps(T))
            error("Could not obtain required number of electrons by filling every state. " *
                  "Increase n_bands.")
        end

        εF = compute_fermi_level(basis, eigenvalues, algorithm; temperature, smearing)

        # Sanity check on Fermi level
        if deviation_n_electrons(basis, eigenvalues, εF; smearing, temperature) < sqrt(eps(T))
            if iszero(temperature)
                error("Unable to find non-fractional occupations that have the " *
                      "correct number of electrons. You should add a temperature.")
            else
                error("This should not happen, debug me!")
            end
        end
    end
    occupation = compute_occupation(basis, eigenvalues, εF; temperature, smearing)

    (; occupation, εF)
end


"""Compute the occupations, given eigenenergies and a Fermi level"""
function compute_occupation(basis::PlaneWaveBasis{T}, eigenvalues, εF::Number;
                            temperature=basis.model.temperature,
                            smearing=basis.model.smearing) where {T}
    # This is needed to get the right behaviour for special floating-point types
    # such as intervals.
    inverse_temperature = iszero(temperature) ? T(Inf) : 1/temperature

    filled_occ = filled_occupation(basis.model)
    occupation = map(eigenvalues) do εk
        occ = filled_occ * Smearing.occupation.(smearing, (εk .- εF) .* inverse_temperature)
        to_device(basis.architecture, occ)
    end
    (; occupation, εF)
end


function compute_fermi_level(basis::PlaneWaveBasis{T}, eigenvalues, ::ZeroTemperature;
                             temperature, smearing)
    # Sanity check that we can indeed fill the appropriate number of states
    if n_electrons % (n_spin * filled_occ) != 0
        error("$n_electrons electrons cannot be attained by filling states with " *
              "occupation $filled_occ. Typically this indicates that you need to put " *
              "a temperature or switch to a calculation with collinear spin polarization.")
    end
    n_fill = div(n_electrons, n_spin * filled_occ, RoundUp)

    # For zero temperature, two cases arise: either there are as many bands
    # as electrons, in which case we set εF to the highest energy level
    # reached, or there are unoccupied conduction bands and we take
    # εF as the midpoint between valence and conduction bands.
    if n_fill == length(eigenvalues[1])
        εF = max_ε
    else
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
end

function compute_fermi_level(basis::PlaneWaveBasis{T}, eigenvalues ::Bisection;
                             temperature, smearing) where {T}
    if iszero(temperature)
        return compute_fermi_level(basis, eigenvalues, ZeroTemperature();
                                   temperature, smearing)
    end

    # Get rough bounds to bracket εF
    min_ε = minimum(minimum, eigenvalues) - 1
    min_ε = mpi_min(min_ε, basis.comm_kpts)
    max_ε = maximum(maximum, eigenvalues) + 1
    max_ε = mpi_max(max_ε, basis.comm_kpts)

    objective(εF) = deviation_n_electrons(basis, eigenvalues, εF; smearing, temperature)
    @assert objective(min_ε) < 0 < objective(max_ε)
    Roots.find_zero(objective, (min_ε, max_ε), Roots.Bisection(), atol=eps(T))
end

function compute_fermi_level(basis::PlaneWaveBasis{T}, eigenvalues ::GaussianNewton;
                             temperature, smearing) where {T}
    εF_guess = compute_fermi_level(basis, eigenvalues, Bisection();
                                   temperature, smearing=Smearing.Gaussian())
    if iszero(temperature) || smearing isa Smearing.Gaussian
        return εF_guess
    end

    n_elec(εF)  = deviation_n_electrons(basis, eigenvalues, εF; smearing, temperature)
    dn_elec(εF) = ForwardDiff.derivative(objective, εF)
    Roots.find_zero((n_elec, dn_elec), εF_guess, Newton())
end
