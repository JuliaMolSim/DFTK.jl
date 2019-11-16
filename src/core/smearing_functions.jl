using SpecialFunctions: erf, factorial

# Implemented smearing functions
# In this x is the ratio (ε .- εF) ./ temperature

# Ref for the equations:
#    - M. Methfessel, A. T. Paxton 1989
#       "High-precision sampling for Brillouin-zone integration in metals"
#     - E. Cancès, V. Ehrlacher, D. Gontier, A. Levitt, D. Lombardi
#       "Numerical quadrature in the brillouin zone for periodic schrodinger operators"
# TODO: Marzari-Vanderbilt "cold smearing"
smearing_fermi_dirac(x) = 1 / (1 + exp(x))
smearing_gaussian(x) = (1 - erf(x)) / 2

H1(x) = 2x
H2(x) = 4x^2 - 2
H3(x) = 8x^3 - 12x
A_coeff(n, T=Float64) = (-1)^n / (factorial(n) * 4^n * sqrt(T(π)))
## TODO switch that to arbitrary precision
const A1 = A_coeff(1)
const A2 = A_coeff(2)
smearing_methfessel_paxton_1(x) = smearing_gaussian(x) + A1 * H1(x) * exp(-x^2)
smearing_methfessel_paxton_2(x) = smearing_methfessel_paxton_1(x) + A2 * H3(x) * exp(-x^2)

# List of available smearing functions
smearing_functions = (smearing_fermi_dirac, smearing_gaussian, smearing_methfessel_paxton_1,
                      smearing_methfessel_paxton_2)
