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

abstract type FermiAlgorithm end

"""Default selection of a Fermi level determination algorithm"""
default_fermialg(::Type{<: Smearing.SmearingFunction}) = FermiTwoStage()
default_fermialg(::Type{Smearing.Gaussian})   = FermiBisection()  # Monotonic smearing
default_fermialg(::Type{Smearing.FermiDirac}) = FermiBisection()  # Monotonic smearing
default_fermialg(::T) where {T <: Smearing.SmearingFunction} = default_fermialg(T)
default_fermialg(model::Model) = default_fermialg(model.smearing)
default_fermialg(basis::AbstractBasis) = default_fermialg(basis.model)

function excess_n_electrons(basis::PlaneWaveBasis, eigenvalues, εF; smearing, temperature)
    occupation = compute_occupation(basis, eigenvalues, εF;
                                    smearing, temperature).occupation
    weighted_ksum(basis, sum.(occupation)) - basis.model.n_electrons
end

"""Compute occupation given eigenvalues and Fermi level"""
function compute_occupation(basis::PlaneWaveBasis{T}, eigenvalues::AbstractVector, εF::Number;
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

"""Compute occupation and Fermi level given eigenvalues and using `fermialg`.
The `tol_n_elec` gives the accuracy on the electron count which should be at least achieved.
"""
function compute_occupation(basis::PlaneWaveBasis{T}, eigenvalues::AbstractVector,
                            fermialg::FermiAlgorithm;
                            tol_n_elec=default_occupation_threshold(),
                            temperature=basis.model.temperature,
                            smearing=basis.model.smearing) where {T}
    if !isnothing(basis.model.εF)  # fixed Fermi level
        εF = basis.model.εF
    else  # fixed n_electrons
        # Check there are enough bands
        max_n_electrons = (filled_occupation(basis.model)
                           * weighted_ksum(basis, length.(eigenvalues)))
        if max_n_electrons < basis.model.n_electrons - tol_n_elec
            error("Could not obtain required number of electrons by filling every state. " *
                  "Increase n_bands.")
        end

        @timing "compute_fermi_level" begin
            εF = compute_fermi_level(basis, eigenvalues, fermialg;
                                     tol_n_elec, temperature, smearing)
        end

        excess  = excess_n_electrons(basis, eigenvalues, εF; smearing, temperature)
        dexcess = ForwardDiff.derivative(εF) do εF
            excess_n_electrons(basis, eigenvalues, εF; smearing, temperature)
        end

        if abs(excess) > tol_n_elec
            @warn("Large deviation of electron count in compute_occupation. (excess=$excess). " *
                  "This may lead to an unphysical solution. Try decreasing the temperature " *
                  "or using a different smearing function.")
        end
        if dexcess < -sqrt(eps(T))
            @warn("Negative electron versus Fermi level derivative in compute_occupation. " *
                  "This may lead to an unphysical solution. Try decreasing the temperature " *
                  "or using a different smearing function.")
        end
    end
    compute_occupation(basis, eigenvalues, εF; temperature, smearing)
end


struct FermiBisection <: FermiAlgorithm end
function compute_fermi_level(basis::PlaneWaveBasis{T}, eigenvalues, ::FermiBisection;
                             temperature, smearing, tol_n_elec) where {T}
    if iszero(temperature)
        return compute_fermi_level(basis, eigenvalues, FermiZeroTemperature();
                                   temperature, smearing, tol_n_elec)
    end

    # Get rough bounds to bracket εF
    min_ε = minimum(minimum, eigenvalues) - 1
    min_ε = mpi_min(min_ε, basis.comm_kpts)
    max_ε = maximum(maximum, eigenvalues) + 1
    max_ε = mpi_max(max_ε, basis.comm_kpts)

    excess(εF) = excess_n_electrons(basis, eigenvalues, εF; smearing, temperature)
    @assert excess(min_ε) < 0 < excess(max_ε)
    εF = Roots.find_zero(excess, (min_ε, max_ε), Roots.Bisection(), atol=eps(T))
    abs(excess(εF)) > tol_n_elec && error("This should not happen ...")
    εF
end


"""
Two-stage Fermi level finding algorithm starting from a Gaussian-smearing guess.
"""
struct FermiTwoStage <: FermiAlgorithm end
function compute_fermi_level(basis::PlaneWaveBasis{T}, eigenvalues, ::FermiTwoStage;
                             temperature, smearing, tol_n_elec) where {T}
    if iszero(temperature)
        return compute_fermi_level(basis, eigenvalues, FermiZeroTemperature();
                                   temperature, smearing, tol_n_elec)
    end

    # Compute a guess using Gaussian smearing
    εF = compute_fermi_level(basis, eigenvalues, FermiBisection();
                             temperature, tol_n_elec, smearing=Smearing.Gaussian())

    # Improve using a Newton-type method in two stages
    excess(εF)  = excess_n_electrons(basis, eigenvalues, εF; smearing, temperature)

    # TODO Could try to use full Newton here instead of Secant, but the combination
    #      Newton + bracketing method seems buggy in Roots
    Roots.find_zero(excess, εF, Roots.Secant(), Roots.Bisection(); atol=eps(T))
end


# Note: This is not exported, but only called by the above algorithms for
# the zero-temperature case.
struct FermiZeroTemperature <: FermiAlgorithm end
function compute_fermi_level(basis::PlaneWaveBasis{T}, eigenvalues, ::FermiZeroTemperature;
                             temperature, smearing, tol_n_elec) where {T}
    filled_occ  = filled_occupation(basis.model)
    n_electrons = basis.model.n_electrons
    n_spin = basis.model.n_spin_components
    @assert iszero(temperature)

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
        εF = maximum(maximum, eigenvalues) + 1
        εF = mpi_max(εF, basis.comm_kpts)
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

    excess(εF) = excess_n_electrons(basis, eigenvalues, εF; temperature, smearing)
    if abs(excess(εF)) > tol_n_elec
        error("Unable to find non-fractional occupations that have the " *
              "correct number of electrons. You should add a temperature.")
    end

    εF
end
