@doc raw"""
    hankel(r, r2_f, l, q)

Compute the Hankel transform
```math
    H[f] = 4\pi \int_0^\infty r f(r) j_l(qr) r dr.
```
The integration is performed by trapezoidal quadrature, and the function takes
as input the radial grid `r`, the precomputed quantity r²f(r) `r2_f`, angular
momentum / spherical bessel order `l`, and the Hankel coordinate `q`.
"""
function hankel(r::AbstractVector, r2_f::AbstractVector, l::Integer,
                q::T)::T where {T<:Real}
    integrand = r2_f .* sphericalbesselj_fast.(l, q .* r)
    return 4T(π) * simpson(r, integrand)
end
