import Roots
#import MPI

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
                            tol_n_elec=default_occupation_threshold(T),
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
Discrete bisection method to find εF in the array εFs such that 
|excess_funcion(εF)| < tol_n_elec 
where excess_function is an increasing function and εFs is a sorted array (increasing).
"""
function discrete_find_zero(excess_function, εFs, tol_n_elec)
    i_min = 1
    i_max = length(εFs)
    excess_min = excess_function(εFs[1])
    excess_max = excess_function(last(εFs))
    if excess_max <= tol_n_elec         # Try to fill all the bands
        εF = last(εFs)
        if excess_max < -tol_n_elec
            error("Could not obtain required number of electrons by filling every state. " *
                    "Increase n_bands.")
        end
    elseif excess_min >= -tol_n_elec    # Try to fill only the smallest band(s)
        εF = first(εFs)
    else                                # Bissection to find first(εFs) < εF < last(εFs)
        while i_max - i_min > 1
            i = div(i_min+i_max, 2)
            εF = εFs[i]
            excess = excess_function(εF)
            if excess < -tol_n_elec
                i_min = i
            elseif excess > tol_n_elec
                i_max = i
            else 
                i_min = i
                i_max = i
            end
        end
    end
    εF
end

# Note: This is not exported, but only called by the above algorithms for
# the zero-temperature case.
struct FermiZeroTemperature <: AbstractFermiAlgorithm end
function compute_fermi_level(basis::PlaneWaveBasis{T}, eigenvalues, ::FermiZeroTemperature;
                             temperature, smearing, tol_n_elec) where {T}
    filled_occ  = filled_occupation(basis.model)
    n_electrons = basis.model.n_electrons
    @assert iszero(temperature)

    # Sanity check that we can indeed fill the appropriate number of states
    if n_electrons % (filled_occ) != 0
        error("$n_electrons electrons cannot be attained by filling states with " *
              "occupation $filled_occ. Typically this indicates that you need to put " *
              "a temperature or switch to a calculation with collinear spin polarization.")
    end

    # Gather the eigenvalues distributed over the kpoints in MPI
    n_bands = length(eigenvalues[1])   # assuming that the same number of bands 
                                       # are computed for each kpoint
    counts = [ Int32(n_bands*sum(length.(basis.krange_allprocs[rank]))) 
               for rank in 1:MPI.Comm_size(basis.comm_kpts) ]
    all_eigenvalues = Array{T}(undef, sum(counts))
    all_eigenvalues_vbuf = MPI.VBuffer(all_eigenvalues, counts)
    MPI.Allgatherv!(reduce(vcat, eigenvalues), all_eigenvalues_vbuf, basis.comm_kpts)
    
    # Search for the Fermi level in between the eigenvalues
    sort!((all_eigenvalues))
    εFs = all_eigenvalues[1:end-1] .+ T(1/2)*diff(all_eigenvalues) # Candidates Fermi-levels
    # Remove candidate Fermi levels that are between two identical eigenvalues
    # (at machine precision)
    εFs = εFs[ diff(all_eigenvalues) .> 2*eps(T)*all_eigenvalues[1:end-1] ]
    push!(εFs, last(all_eigenvalues) + T(1))
    
    excess_function = εF->excess_n_electrons(basis, eigenvalues, εF; temperature, smearing)
    εF = discrete_find_zero(excess_function, εFs, tol_n_elec)
    
    occ = compute_occupation(basis, eigenvalues, εF; temperature, smearing).occupation
    merged_spin_occupations = sum(  occ[krange_spin(basis, i)] 
                                    for i=1:basis.model.n_spin_components )
    if !allequal(merged_spin_occupations)
        @warn("Not all kpoints have the same number of occupied states, which could mean "*
              "that a metallic system is treated at zero temperature.")
    end
    excess = excess_n_electrons(basis, eigenvalues, εF; temperature, smearing)
    if abs(excess) > tol_n_elec
        error("Unable to find non-fractional occupations that have the " *
              "correct number of electrons. You should add a temperature.")
    end
    εF
end

"""
Check that all orbitals are fully occupied.
"""
function check_full_occupation(basis::PlaneWaveBasis, occupation)
    filled_occ = filled_occupation(basis.model)
    for occ_k in occupation
        all(occ_k .== filled_occ) || error("Only full occupation is supported, but $occ_k has partial occupation.")
    end
end
