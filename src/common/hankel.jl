@doc raw"""
    hankel(r, r2_f, l, p)

Compute the **regularized** Hankel transform
```math
    H[f] = \frac{1}{p^l} ⋅ 4\pi \int_0^\infty r f(r) j_l(p·r) r dr.
```
The integration is performed by simpson quadrature, and the function takes
as input the radial grid `r`, the precomputed quantity r²f(r) `r2_f`, angular
momentum / spherical bessel order `l`, and the Hankel coordinate `p`.

The regularization by 1/p^l avoids numerical issues for small p:
instead of computing the unregularized Hankel transform times a spherical harmonic,
we can compute the regularized Hankel transform times a solid harmonic.
"""
function hankel(r::AbstractVector, r2_f::AbstractVector, l::Integer, p::T)::T where {T<:Real}
    quadrature = default_psp_quadrature(r)
    hankel(quadrature, r, r2_f, l, p)
end

function hankel(quadrature, r::AbstractVector, r2_f::AbstractVector, l::Integer, p::T)::T where {T<:Real}
    @assert length(r) == length(r2_f)
    if abs(p) <= 10 * eps(T)
        # Use j_l(x) ≈ x^l / (2l+1)!! for small p
        l == 0 && return 4T(π) * quadrature((i, ri) -> r2_f[i], r)
        l == 1 && return 4T(π) * quadrature((i, ri) -> r2_f[i] * ri, r) / 3
        l == 2 && return 4T(π) * quadrature((i, ri) -> r2_f[i] * ri^2, r) / 15
        l == 3 && return 4T(π) * quadrature((i, ri) -> r2_f[i] * ri^3, r) / 105
        l == 4 && return 4T(π) * quadrature((i, ri) -> r2_f[i] * ri^4, r) / 945
        l == 5 && return 4T(π) * quadrature((i, ri) -> r2_f[i] * ri^5, r) / 10395
        throw(BoundsError()) # specific l not implemented
    end
    1/p^l * 4T(π) * quadrature(r) do i, ri
        r2_f[i] * sphericalbesselj_fast(l, p * ri)
    end
end