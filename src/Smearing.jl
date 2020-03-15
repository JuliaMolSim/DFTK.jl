# Smearing functions

# Ref for the equations:
#    - M. Methfessel, A. T. Paxton 1989
#       "High-precision sampling for Brillouin-zone integration in metals"
#     - E. Cancès, V. Ehrlacher, D. Gontier, A. Levitt, D. Lombardi
#       "Numerical quadrature in the brillouin zone for periodic schrodinger operators"
# See also https://www.vasp.at/vasp-workshop/k-points.pdf
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
(f(x) - f(y))/(x - y), computed stably in the case where x and y are close
"""
function occupation_divided_difference(S::SmearingFunction, x, y, εF, T)
    if T == 0
        if x == y
            zero(x)
        else
            fx = x < εF ? 1 : 0
            fy = y < εF ? 1 : 0
            (fx-fy)/(x-y)
        end
    else
        f(z) = occupation(S, (z-εF)/T)
        fp(z) = occupation_derivative(S, (z-εF)/T)/T
        divided_difference_(f, fp, x, y)
    end
end
function divided_difference_(f, fp, x, y)
    # This is only accurate to sqrt(ε)
    ε = eps(typeof(x))
    abs(x-y) < sqrt(ε) && return fp(x)
    (f(x) - f(y))/(x-y)
end

"""
Entropy. Note that this is a function of the energy `x`, not of `occupation(x)`.
This function satisfies s' = x f' (see https://www.vasp.at/vasp-workshop/k-points.pdf
p. 12 and https://arxiv.org/pdf/1805.07144.pdf p. 18.
"""
entropy(S::SmearingFunction, x) = error()

struct None <: SmearingFunction end
occupation(S::None, x) = x > 0 ? zero(x) : one(x)
entropy(S::None, x) = zero(x)

function xlogx(x)
    iszero(x) ? zero(x) : x * log(x)
end

struct FermiDirac <: SmearingFunction end
occupation(S::FermiDirac, x) = 1 / (1 + exp(x))
# entropy(f) = -(f log f + (1-f)log(1-f)), where f = 1/(1+exp(x))
# this "simplifies" to -(x*exp(x)/(1+exp(x)) - log(1+exp(x)))
# although that is not especially useful...
function entropy(S::FermiDirac, x)
    f = occupation(S, x)
    - (xlogx(f) + xlogx(1 - f))
end

struct Gaussian <: SmearingFunction end
occupation(S::Gaussian, x) = (1 - erf(x)) / 2
entropy(S::Gaussian, x) = 1 / (2sqrt(typeof(x)(pi))) * exp(-x^2)

H1(x) = 2x
H2(x) = 4x^2 - 2
H3(x) = 8x^3 - 12x
H4(x) = 16x^4 - 48x^2 + 12
A(n, T=Float64) = (-1)^n / (factorial(n) * 4^n * sqrt(T(π)))

struct MethfesselPaxton1 <: SmearingFunction end
function occupation(S::MethfesselPaxton1, x)
    x == Inf && return zero(x)
    x == -Inf && return one(x)
    occupation(Gaussian(), x) + A(1, typeof(x))*H1(x)*exp(-x^2)
end
entropy(S::MethfesselPaxton1, x) = 1/2 * A(1, typeof(x)) * H2(x) * exp(-x^2)

struct MethfesselPaxton2 <: SmearingFunction end
function occupation(S::MethfesselPaxton2, x)
    x == Inf && return zero(x)
    x == -Inf && return one(x)
    occupation(MethfesselPaxton1(), x) + A(2, typeof(x))*H3(x)*exp(-x^2)
end
entropy(S::MethfesselPaxton2, x) = 1/2 * A(2, typeof(x)) * H4(x) * exp(-x^2)

# TODO: Marzari-Vanderbilt "cold smearing"

# List of available smearing functions
smearing_methods = (None, FermiDirac, Gaussian, MethfesselPaxton1, MethfesselPaxton2)

# these are not broadcastable
import Base.Broadcast.broadcastable
Base.Broadcast.broadcastable(S::SmearingFunction) = Ref(S)
end
