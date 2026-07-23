@doc raw"""
    sphericalbesselj_fast(l::Integer, x, (sinx, cosx)=sincos(x))

Returns the spherical Bessel function of the first kind ``j_l(x)``. Consistent with
[Wikipedia](https://en.wikipedia.org/wiki/Bessel_function#Spherical_Bessel_functions) and
with `SpecialFunctions.sphericalbesselj`. Specialized for integer ``0 ≤ l ≤ 5``.

Warning: For small arguments, the current implementation is not numerically stable.
"""
@fastmath function sphericalbesselj_fast(l::Integer, x::T, (s, c)=sincos(x))::T where {T}
    iszero(x) && return T(l == 0)  # jₗ(0) = δₗ₀
    # TODO: fix numerical stability for small x
    l == 0 && return s / x
    l == 1 && return (s - c * x) / x^2
    l == 2 && return (s * (3 - x^2) - 3c * x) / x^3
    l == 3 && return (s * (15 - 6x^2) + c * (x^3 - 15x)) / x^4
    l == 4 && return (s * (105 - 45x^2 + x^4) + c * (10x^3 - 105x)) / x^5
    l == 5 && return (s * (945 - 420x^2 + 15x^4) + c * (-945x + 105x^3 - x^5)) / x^6
    throw(BoundsError()) # specific l not implemented
end
