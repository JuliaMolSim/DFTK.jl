import SpecialFunctions

"""
    sphericalbesselj(::Val{nu}, x::Number)

Returns the spherical Bessel function of the first kind j_nu(x). Consistent with
https://en.wikipedia.org/wiki/Bessel_function#Spherical_Bessel_functions.
"""
@inline @fastmath function sphericalbesselj_fast(::Val{0}, x::T)::T where {T}
    iszero(x) && return one(T)
    sin(x) / x
end

@inline @fastmath function sphericalbesselj_fast(::Val{1}, x::T)::T where {T}
    iszero(x) && return zero(T)
    (sin(x) - cos(x) * x) / x^2
end

@inline @fastmath function sphericalbesselj_fast(::Val{2}, x::T)::T where {T}
    iszero(x) && return zero(T)
    (sin(x) * (3 - x^2) + cos(x) * (-3x)) / x^3
end

@inline @fastmath function sphericalbesselj_fast(::Val{3}, x::T)::T where {T}
    iszero(x) && return zero(T)
    (sin(x) * (15 - 6x^2) + cos(x) * (x^3 - 15x)) / x^4
end

@inline @fastmath function sphericalbesselj_fast(::Val{4}, x::T)::T where {T}
    iszero(x) && return zero(T)
    (sin(x) * (105 - 45x^2 + x^4) + cos(x) * (10x^3 - 105x)) / x^5
end

@inline @fastmath function sphericalbesselj_fast(::Val{5}, x::T)::T where {T}
    iszero(x) && return zero(T)
    (sin(x) * (945 - 420x^2 + 15x^4) + cos(x) * (-945x + 105x^3 - x^5)) / x^6
end

@inline function sphericalbesselj_fast(::Val{nu}, x::T)::T where {T, nu}
    SpecialFunctions.sphericalbesselj(nu, x)
end