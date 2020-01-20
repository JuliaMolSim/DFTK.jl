# Implemented smearing functions

# Ref for the equations:
#    - M. Methfessel, A. T. Paxton 1989
#       "High-precision sampling for Brillouin-zone integration in metals"
#     - E. Cancès, V. Ehrlacher, D. Gontier, A. Levitt, D. Lombardi
#       "Numerical quadrature in the brillouin zone for periodic schrodinger operators"
# See also https://www.vasp.at/vasp-workshop/k-points.pdf
# TODO: Marzari-Vanderbilt "cold smearing"
module Smearing

using SpecialFunctions: erf, factorial
using ForwardDiff

abstract type SmearingFunction end
"""
Occupation at `x`, where in practice x = (ε - εF) / T.
"""
occupation(S::SmearingFunction, x) = error()
"""
Derivative of the occupation function, approximation to minus the delta function.
"""
occupation_derivative(S::SmearingFunction, x) = ForwardDiff.derivative(x -> occupation(S, x), x)
"""
Entropy. Note that this is a function of the energy `x`, not of `occupation(x)`.
"""
entropy(S::SmearingFunction, x) = error()

struct None <: SmearingFunction end
occupation(S::None, x) = x > 0 ? zero(x) : one(x)
entropy(S::None, x) = zero(x)

struct FermiDirac <: SmearingFunction end
occupation(S::FermiDirac, x) = 1 / (1 + exp(x))
# entropy(f) = -(f log f + (1-f)log(1-f)), where f = 1/(1+exp(x)), which "simplifies" to
entropy(S::FermiDirac, x) = -(x*exp(x)/(1+exp(x)) - log(1+exp(x)))
# Sanity check:
begin
    x = .123
    f = occupation(FermiDirac(), x)
    e = -(f*log(f) + (1-f)*log(1-f))
    @assert entropy(FermiDirac(), x) ≈ e
end

struct Gaussian <: SmearingFunction end
occupation(S::Gaussian, x) = (1 - erf(x)) / 2
entropy(S::Gaussian, x) = 1/(2sqrt(typeof(x)(pi)))*exp(-x^2)

H1(x) = 2x
H2(x) = 4x^2 - 2
H3(x) = 8x^3 - 12x
H4(x) = 16x^4 - 48x^2 + 12
A_coeff(n, T=Float64) = (-1)^n / (factorial(n) * 4^n * sqrt(T(π)))
## TODO switch that to arbitrary precision
const A1 = A_coeff(1)
const A2 = A_coeff(2)
struct MethfesselPaxton1 <: SmearingFunction end
function occupation(S::MethfesselPaxton1, x)
    if x == Inf
        return zero(x)
    elseif x == -Inf
        return one(x)
    end
    occupation(Gaussian(), x) + A1 * H1(x) * exp(-x^2)
end
entropy(S::MethfesselPaxton1, x) = 1/2*A1*H2(x)*exp(-x^2)

struct MethfesselPaxton2 <: SmearingFunction end
function occupation(S::MethfesselPaxton2, x)
    if x == Inf
        return zero(x)
    elseif x == -Inf
        return one(x)
    end
    occupation(MethfesselPaxton1(), x) + A2 * H3(x) * exp(-x^2)
end
entropy(S::MethfesselPaxton2, x) = 1/2*A2*H4(x)*exp(-x^2)

# List of available smearing functions
smearing_methods = (None, FermiDirac, Gaussian, MethfesselPaxton1, MethfesselPaxton2)

# these are not broadcastable
import Base.Broadcast.broadcastable
Base.Broadcast.broadcastable(S::SmearingFunction) = Ref(S)
end
