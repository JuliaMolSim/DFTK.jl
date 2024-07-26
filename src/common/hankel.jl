@doc raw"""
    hankel(r, r2_f, l, p)

Compute the Hankel transform
```math
    H[f] = 4\pi \int_0^\infty r f(r) j_l(p·r) r dr.
```
The integration is performed by simpson quadrature, and the function takes
as input the radial grid `r`, the precomputed quantity r²f(r) `r2_f`, angular
momentum / spherical bessel order `l`, and the Hankel coordinate `p`.
"""
function hankel(r::AbstractVector, r2_f::AbstractVector, l::Integer, p::T)::T where {T<:Real}
    @assert length(r) == length(r2_f)
    4T(π) * simpson(r) do i, ri
        r2_f[i] * sphericalbesselj_fast(l, p * ri)
    end
end
