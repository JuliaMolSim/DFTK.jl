import Roots
using SpecialFunctions


"""
Compute the occupation at zero temperature and without smearing
for `n_elec` electrons and the bands `Psi` with associated `energies`.
"""
function occupation_zero_temperature(basis, energies, Psi, n_elec)
    n_bands = size(Psi[1], 2)
    T = eltype(energies[1])

    @assert n_elec % 2 == 0 "Only even number of electrons implemented"
    @assert n_bands ≥ n_elec / 2

    occupation = similar(basis.kpoints, Vector{T})
    HOMO = -Inf
    LUMO = Inf
    for ik in 1:length(occupation)
        occupation[ik] = zeros(T, n_bands)
        occupation[ik][1:div(n_elec,2)] .= 2
        HOMO = max(HOMO, energies[ik][div(n_elec,2)])
        LUMO = min(LUMO, energies[ik][div(n_elec,2)+1])
    end
    ## We just error in the case of metal; the proper thing to do here
    ## is to do a bisection for the Fermi level as in the case of
    ## finite-temperature (although that has a dreadful convergence
    ## rate)
    @assert HOMO ≤ LUMO
    occupation
end

# Ref for the equations: Methfessel-Paxton paper, or
# E. Cancès, V. Ehrlacher, D. Gontier, A. Levitt, D. Lombardi,
# Numerical quadrature in the brillouin zone for periodic schrodinger
# operators
# TODO: Marzari-Vanderbilt "cold smearing"
smearing_fermi_dirac(x) = 1/(1+exp(x))
smearing_gaussian(x) = 1/2*(1-erf(x))

H1(x) = 2x
H2(x) = 4x^2 - 2
H3(x) = 8x^3 - 12x
A_coeff(n) = (-1)^n/(factorial(n)*4^n*sqrt(pi))
const A1 = A_coeff(1); const A2 = A_coeff(2);
smearing_methfessel_paxton_1(x) = smearing_gaussian(x) + A1*H1(x)*exp(-x^2)
smearing_methfessel_paxton_2(x) = smearing_gaussian(x) + A1*H1(x)*exp(-x^2) + A2*H3(x)*exp(-x^2)

# List of available smearing functions, useful in tests
smearing_functions = (smearing_fermi_dirac, smearing_gaussian, smearing_methfessel_paxton_1, smearing_methfessel_paxton_2)

""" Compute the occupation `f(ε) = smearing((ε-εF)/T)`, for `n_elec`
electrons and the bands `Psi` with associated `energies`."""
function occupation_temperature(basis, energies, Psi, n_elec, T=0, smearing=smearing_fermi_dirac)
    n_bands = size(Psi[1], 2)

    @assert n_elec % 2 == 0 "Only even number of electrons implemented"
    @assert n_bands ≥ n_elec / 2
    @assert sum(basis.kweights) ≈ 1

    compute_occupation(εF) = [2*smearing.((ε .- εF)./T) for ε in energies]
    compute_n_elec(εF) = sum(basis.kweights .* sum.(compute_occupation(εF)))

    min_ε = minimum([minimum(ε) for ε in energies])-1
    max_ε = maximum([maximum(ε) for ε in energies])+1
    @assert compute_n_elec(min_ε) < n_elec < compute_n_elec(max_ε)
    # Just use bisection here; note that with MP smearing there might
    # be multiple possible Fermi levels. This could be sped up with more
    # advanced methods (eg false position), but more care has to be
    # taken with convergence criteria and the like
    εF = Roots.find_zero(εF -> compute_n_elec(εF) - n_elec, (min_ε, max_ε), Roots.Bisection())
    @assert compute_n_elec(εF) ≈ n_elec
    compute_occupation(εF)
end
