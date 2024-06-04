@doc raw"""
    hankel(r, r2_f, l, p)

Compute the Hankel transform
```math
    H[f] = 4\pi \int_0^\infty r f(r) j_l(p·r) r dr.
```
The integration is performed by trapezoidal quadrature, and the function takes
as input the radial grid `r`, the precomputed quantity r²f(r) `r2_f`, angular
momentum / spherical bessel order `l`, and the Hankel coordinate `p`.
"""
function hankel(r::AbstractVector, r2_f::AbstractVector, l::Integer, p::T)::T where {T<:Real}
    @assert length(r) == length(r2_f)
    4T(π) * simpson(r, r2_f) do i, ri, r2_f
        r2_f[i] * sphericalbesselj_fast(l, p * ri)
    end
end

function hankel(r::AbstractVector, r2_f::AbstractVector, l::Integer, p::TT) where {TT <: ForwardDiff.Dual}
    # This custom rule uses two properties of the hankel transform:
    #   d H[f] / dp = 4\pi \int_0^∞ r^2 f(r) j_l'(p⋅r)⋅r dr
    # and that
    #   j_l'(x) = l / x * j_l(x) - j_{l+1}(x)
    # and tries to avoid allocations as much as possible, which hurt in this inner loop.
    #
    # One could implement this by custom rules in integration and spherical bessels, but
    # the tricky bit is to exploit that one needs both the j_l'(p⋅r) and j_l(p⋅r) values
    # but one does not want to precompute and allocate them into arrays
    # TODO Investigate custom rules for bessels and integration

    T  = ForwardDiff.valtype(TT)
    pv = ForwardDiff.value(p)

    jl = sphericalbesselj_fast.(l, pv .* r)
    value = 4T(π) * simpson((i, r, r2_f, jl) -> r2_f[i] * jl[i], r, r2_f, jl)

    if iszero(pv)
        return TT(value, zero(T) * ForwardDiff.partials(p))
    end

    function derivative_value(i, r, r2_f, jl, pv)
        (r2_f[i] * (l * jl[i] / pv - r .* sphericalbesselj_fast(l+1, pv * r)))
    end
    derivative = 4T(π) * simpson(derivative_value, r, r2_f, jl, pv)
    TT(value, derivative * ForwardDiff.partials(p))
end
