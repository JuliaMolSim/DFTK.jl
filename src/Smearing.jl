# Smearing functions

# Ref for the equations:
#    - M. Methfessel, A. T. Paxton 1989
#       "High-precision sampling for Brillouin-zone integration in metals"
#     - N. Marzari, D. Vanderbilt, A. De Vita, M. C. Payne
#       "Thermal Contraction and Disordering of the Al(110) Surface"
#     - E. Cancès, V. Ehrlacher, D. Gontier, A. Levitt, D. Lombardi
#       "Numerical quadrature in the brillouin zone for periodic schrodinger operators"
# See also https://www.vasp.at/vasp-workshop/k-points.pdf
module Smearing

using SpecialFunctions: erf, erfc, factorial
import ForwardDiff

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
function occupation_divided_difference(S::SmearingFunction, x, y, εF, temp)
    if temp == 0
        if x == y
            zero(x)
        else
            fx = x < εF ? 1 : 0
            fy = y < εF ? 1 : 0
            (fx-fy)/(x-y)
        end
    else
        f(z) = occupation(S, (z-εF) / temp)
        fder(z) = occupation_derivative(S, (z-εF)/temp) / temp
        divided_difference_(f, fder, x, y)
    end
end
function divided_difference_(f, fder, x::T, y) where T
    # (f(x) - f(y))/(x - y) is accurate to ε/|x-y|
    # so for x ~= y we use the approximation (f'(x)+f'(y))/2,
    # which is accurate to |x-y|^2, and therefore better when |x-y| ≤ cbrt(ε)
    # The resulting method is accurate to ε^2/3
    abs(x-y) < cbrt(eps(T)) && return (fder(x) + fder(y))/2
    (f(x)-f(y)) / (x-y)
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

struct FermiDirac <: SmearingFunction end
occupation(S::FermiDirac, x) = 1 / (1 + exp(x))
function xlogx(x)
    iszero(x) ? zero(x) : x * log(x)
end
function entropy(S::FermiDirac, x)
    f = occupation(S, x)
    - (xlogx(f) + xlogx(1 - f))
end
function occupation_divided_difference(S::FermiDirac, x, y, εF, temp)
    temp == 0 && return occupation_divided_difference(None(), x, y, εF, temp)
    f(z) = occupation(S, (z-εF) / temp)
    fder(z) = occupation_derivative(S, (z-εF)/temp) / temp
    # For a stable computation we use
    # (fx - fy) = fx fy (exp(y) - exp(x)) = fx fy exp(x) expm1(y-x)
    # which we symmetrize. This can overflow, in which case
    # we fall back to the standard method
    large_float = floatmax(typeof(x)) / 1e4 # conservative
    will_exp_overflow(z1, z2) = abs((z1-z2)/temp) > log(large_float)
    if x == y || will_exp_overflow(x, y) || will_exp_overflow(x, εF) || will_exp_overflow(y, εF)
        divided_difference_(f, fder, x, y)
    else
        Δfxy = f(x) * f(y) * exp((x-εF)/temp) * expm1((y-x)/temp)
        Δfyx = f(x) * f(y) * exp((y-εF)/temp) * expm1((x-y)/temp)
        (Δfxy-Δfyx) / 2 / (x-y)
    end
end

struct Gaussian <: SmearingFunction end
occupation(S::Gaussian, x) = erfc(x) / 2
entropy(S::Gaussian, x::T) where T = 1 / (2 * sqrt(T(π))) * exp(-x^2)

# NB: the Fermi energy with Marzari-Vanderbilt smearing is __not__ unique
struct MarzariVanderbilt <: SmearingFunction end
function occupation(S::MarzariVanderbilt, x::T) where T
    return (
        -erf(x + 1/sqrt(T(2))) / 2 
        + 1/sqrt(2*T(π)) * exp(-(-x - 1/sqrt(T(2)))^2) + 1/T(2)
    )
end
function entropy(S::MarzariVanderbilt, x::T) where T
    return 1/sqrt(2*T(π)) * (x + 1/sqrt(T(2))) * exp(-(-x - 1/sqrt(T(2)))^2)
end

"""
`A` term in the Hermite delta expansion
"""
A(T, n) = (-1)^n / (factorial(n) * 4^n * sqrt(T(π)))

"""
Standard Hermite function using physicist's convention.
"""
function H(x, n)
    if n < 0
        return zero(x)
    elseif n == 0
        return one(x)
    else
        return 2 * x * H(x, n-1) - 2 * (n-1) * H(x, n-2)
    end
end

# NB: the Fermi energy with Methfessel-Paxton smearing is __not__ unique
struct MethfesselPaxton <: SmearingFunction
    order::Int
end
function occupation(S::MethfesselPaxton, x::T) where T
    x == Inf && return zero(x)
    x == -Inf && return one(x)
    f₀ = erfc(x) / 2  # 0-order Methfessel-Paxton smearing is Gaussian smearing
    Σfₙ = sum(i -> A(T, i) * H(x, 2i - 1), 1:S.order)
    return f₀ + Σfₙ * exp(-x^2)
end
function entropy(S::MethfesselPaxton, x::T) where T
    return sum(i -> A(T, i) * (H(x, 2i) / 2 + 2i * H(x, 2i - 2)), 0:S.order) * exp(-x^2)
end


# these are not broadcastable
import Base.Broadcast.broadcastable
Base.Broadcast.broadcastable(S::SmearingFunction) = Ref(S)
end
