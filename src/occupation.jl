import Roots

# Functions for finding the Fermi level and occupation numbers for bands
# The goal is to find εF so that
# f_i = filled_occ * f((εi-εF)/θ) with θ = temperature
# sum_i f_i = n_electrons
# If temperature is zero, (εi-εF)/θ = ±∞.
# The occupation function is required to give 1 and 0 respectively in these cases.
#
# For monotonic smearing functions (like Gaussian or Fermi-Dirac) we right now just
# use FermiBisection. For non-monototic functions (like MP, MV) with possibly multiple
# Fermi levels we use a two-stage process (FermiTwoStage) trying to avoid non-physical
# Fermi levels (see https://arxiv.org/abs/2212.07988).

abstract type AbstractFermiAlgorithm end

"""Default selection of a Fermi level determination algorithm"""
default_fermialg(::Smearing.SmearingFunction) = FermiTwoStage()
default_fermialg(::Smearing.Gaussian)   = FermiBisection()  # Monotonic smearing
default_fermialg(::Smearing.FermiDirac) = FermiBisection()  # Monotonic smearing
default_fermialg(model::Model)          = default_fermialg(model.smearing)

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
                            fermialg::AbstractFermiAlgorithm=default_fermialg(basis.model);
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

        εF = compute_fermi_level(basis, eigenvalues, fermialg;
                                 tol_n_elec, temperature, smearing)
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
            @warn("Negative density of states (electron count versus Fermi level derivative) " *
                  "encountered in compute_occupation. This may lead to an unphysical " *
                  "solution. Try decreasing the temperature or using a different smearing " *
                  "function.")
        end
    end
    compute_occupation(basis, eigenvalues, εF; temperature, smearing)
end


struct FermiBisection <: AbstractFermiAlgorithm end
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
struct FermiTwoStage <: AbstractFermiAlgorithm end
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

"""
Implementation of the Fermi level determination algorithm suggested
in [arxiv 2212.07988](https://arxiv.org/abs/2212.07988).
"""
@kwdef struct FermiNewton <: FermiAlgorithm
    tol_newton_accept = 1e-2
end

# Note: The reference QuantumEspresso implementation uses tol_n_elec=1e-2, which seems
#       very large. They argue this is fine to reach a more physical Fermi level albeit
#       introducing a mismatch on the electron count. We are more conservative here
#       and use the occupation threshold as a value for orientation.
function compute_fermi_level(basis::PlaneWaveBasis{T}, eigenvalues, fermialg::FermiNewton;
                             temperature, smearing, tol_n_elec) where {T}
    if iszero(temperature)
        return compute_fermi_level(basis, eigenvalues, FermiZeroTemperature();
                                   temperature, smearing, tol_n_elec)
    end

    # Compute a guess using Gaussian smearing
    εF = compute_fermi_level(basis, eigenvalues, FermiBisection();
                             temperature, tol_n_elec, smearing=Smearing.Gaussian())

    excess(εF) = excess_n_electrons(basis, eigenvalues, εF; smearing, temperature)
    abs(excess(εF)) < tol_n_elec && return εF  # Early exit

    # Using the square of the smearing function makes the Newton iterations more reliable.
    # We should additionally square the tolerance, thus limiting the maximally attainable
    # accuracy in the electron count, but this is not done in the original paper, so we
    # are also not doing it here.
    objective(εF)  = abs2(excess(εF))
    function dobjective(εF::T)
        deriv = ForwardDiff.derivative(objective, εF)
        # Force Newton iteration to stop if derivative is small
        return deriv < tol_n_elec ? zero(T) : deriv
    end

    try
        εF = Roots.find_zero((objective, dobjective), εF, Roots.Newton(); atol=tol_n_elec)
    catch e
        (e isa Roots.ConvergenceFailed) || rethrow()
    end
    if abs(excess(εF)) < fermialg.tol_newton_accept
        return εF
    else
        @warn "Newton algorithm failed to determine Fermi level. Falling back to Bisection."
        return compute_fermi_level(basis, eigenvalues, FermiBisection();
                                   temperature, smearing, tol_n_elec)
    end
end


# FermiExtrapolation works by extrapolating the Fermi level found using a monotonic smearing
# (f₀) to the Fermi level of the desired smearing function f₁.
#     g₀(εF) = ∑_i f₀((ε-εF) / T) - N
#     g₁(εF) = ∑_i f₁((ε-εF) / T) - N
#
#     g(εF, α) = (1-α) g₀(εF) + g₁(εF)
#
# The constraint g(εF, α) = 0 defines an implicit function εF(α). By differentiating the
# constraint twice we find εF' and εF''.
#
# 0 = dg/dα = (∂g/∂εF) (∂εF/∂α) + (∂g/∂α) (∂α/∂α)
# thus
#     grad = (∂εF/∂α) = - (∂g/∂εF)^{-1} * (∂g/∂α)
#
# 0 = d²g/d²α =   (∂/∂εF) [(∂g/∂εF) (∂εF/∂α) + (∂g/∂α)] (∂εF/∂α)
#               + (∂/∂α)  [(∂g/∂εF) (∂εF/∂α) + (∂g/∂α)] (∂α/∂α)
#             =   [(∂²g/∂²εF)  (∂εF/∂α) + (∂²g/∂α∂εF)] (∂εF/∂α)
#               + [(∂²g/∂α∂εF) (∂εF/∂α) + (∂g/∂εF) (∂²εF/∂²α) + (∂²g/∂α²)]
#             = [(∂²g/∂²εF) (∂εF/∂α) + 2 (∂²g/∂α∂εF)] (∂εF/∂α) + (∂g/∂εF) (∂²εF/∂²α)
# since (∂²g/∂α²) = 0
# thus
#     hess = (∂²εF/∂²α) = - (∂g/∂εF)^{-1} (∂εF/∂α) [(∂²g/∂²εF) (∂εF/∂α) + 2 (∂²g/∂α∂εF)]
# which makes up a model
#     model(α) = εF₀ + δεF(α) = εF₀ + grad * (α - α₀) + 1/2 hess (α - α₀)^2
#
# To ensure the validity of the second-order model, we check that g(εF(α), α) does not get
# too large.
#
Base.@kwdef struct FermiExtrapolation <: FermiAlgorithm
    maxiter  = 10
    modeltol = 1e-3
    verbose  = false
end
function compute_fermi_level(basis::PlaneWaveBasis{T}, eigenvalues, method::FermiExtrapolation;
                             temperature, smearing, tol_n_elec) where {T}
    if iszero(temperature)
        return compute_fermi_level(basis, eigenvalues, FermiZeroTemperature();
                                   temperature, smearing, tol_n_elec)
    end

    # The excess of electrons function and its derivatives
    function g(α, εF)
        ((1-α) * excess_n_electrons(basis, eigenvalues, εF;
                                    temperature, smearing=Smearing.Gaussian())
         +   α * excess_n_electrons(basis, eigenvalues, εF; smearing, temperature))
    end
    g_ε(α, εF)  = ForwardDiff.derivative(εF ->   g(α, εF), εF)
    g_εε(α, εF) = ForwardDiff.derivative(εF -> g_ε(α, εF), εF)
    g_α(α, εF)  = ForwardDiff.derivative(α  ->   g(α, εF), α)
    g_αε(α, εF) = ForwardDiff.derivative(α  -> g_ε(α, εF), α)

    # Compute a guess using Gaussian smearing
    εF = compute_fermi_level(basis, eigenvalues, FermiBisection();
                             temperature, tol_n_elec, smearing=Smearing.Gaussian())
    abs(g(1.0, εF)) < tol_n_elec && return εF  # Early exit

    αF = 0.0
    for i in 1:method.maxiter
        if method.verbose && mpi_master()
            println("")
            println("-------  Iter $i -- αF=$αF εF=$εF --------")
            println("")
        end

        # Construct model for εF(α)
        grad = - g_α(αF, εF) / g_ε(αF, εF)
        hess = - grad * (g_εε(αF, εF) * grad + 2g_αε(αF, εF)) / g_ε(αF, εF)
        model_εF = α -> εF + grad * (α-αF) + hess * (α-αF)^2 / 2

        # Find a range for which the model is valid
        α_trials = [α for α in range(0.1, 1.0, length=method.maxiter) if α > αF]
        αF = α_trials[1]
        for α_trial in α_trials
            error = g(α_trial,   model_εF(α_trial))
            deriv = g_ε(α_trial, model_εF(α_trial))
            if abs(error) ≥ method.modeltol || deriv ≥ -method.modeltol
                break  # Don't trust model any more
            else
                αF = α_trial
            end
        end

        εF = model_εF(αF)
        if αF < 1.0
            εF = Roots.find_zero(εF -> g(αF, εF), εF, Roots.Order0();
                                 atol=method.modeltol / 10, method.verbose)
        else
            break
        end
    end

    if method.verbose && mpi_master()
        println("")
        println("-------  Finish  εF=$εF --------")
        println("")
    end
    Roots.find_zero(εF -> g(αF, εF), εF, Roots.Order0(); atol=tol_n_elec, method.verbose)
end


# Note: This is not exported, but only called by the above algorithms for
# the zero-temperature case.
struct FermiZeroTemperature <: AbstractFermiAlgorithm end
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
