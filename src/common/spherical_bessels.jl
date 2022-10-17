import SpecialFunctions

"""
    sphericalbesselj(l::Integer, x::Number)

Returns the spherical Bessel function of the first kind j_l(x). Consistent with
https://en.wikipedia.org/wiki/Bessel_function#Spherical_Bessel_functions.
"""
@fastmath function sphericalbesselj_fast(l::Integer, x::T)::T where {T}
    if l == 0
        iszero(x) && return one(T)
        return sin(x) / x
    end

    iszero(x) && return zero(T)

    l == 1 && return (sin(x) - cos(x) * x) / x^2
    l == 2 && return (sin(x) * (3 - x^2) + cos(x) * (-3x)) / x^3
    l == 3 && return (sin(x) * (15 - 6x^2) + cos(x) * (x^3 - 15x)) / x^4
    l == 4 && return (sin(x) * (105 - 45x^2 + x^4) + cos(x) * (10x^3 - 105x)) / x^5
    l == 5 && return (sin(x) * (945 - 420x^2 + 15x^4) + cos(x) * (-945x + 105x^3 - x^5)) / x^6
    l  > 5 && return SpecialFunctions.sphericalbesselj(l, x)
end