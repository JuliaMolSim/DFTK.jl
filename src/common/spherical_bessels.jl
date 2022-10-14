import SpecialFunctions

"""
    sphericalbesselj(::Val{l}, x)

Returns the spherical Bessel function of the first kind j_l(x). Consistent with
https://en.wikipedia.org/wiki/Bessel_function#Spherical_Bessel_functions.
"""
sphericalbesselj(::Val{l}, x) where {l} = SpecialFunctions.sphericalbesselj(l, x)  # Fallback

@inline @fastmath function sphericalbesselj(::Val{0}, x)
    sin(x) / x
end

@inline @fastmath function sphericalbesselj(::Val{1}, x)
    (sin(x) - cos(x) * x) / x^2
end

@inline @fastmath function sphericalbesselj(::Val{2}, x)
    (sin(x) * (3 - x^2) + cos(x) * (-3x)) / x^3
end

@inline @fastmath function sphericalbesselj(::Val{3}, x)
    (sin(x) * (15 - 6x^2) + cos(x) * (x^3 - 15x)) / x^4
end

@inline @fastmath function sphericalbesselj(::Val{4}, x)
    (sin(x) * (105 - 45x^2 + x^4) + cos(x) * (10x^3 - 105x)) / x^5
end

@inline @fastmath function sphericalbesselj(::Val{5}, x)
    (sin(x) * (945 - 420x^2 + 15x^4) + cos(x) * (-945x + 105x^3 - x^5)) / x^6
end
