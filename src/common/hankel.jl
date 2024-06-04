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
    4T(π) * simpson(r) do i, ri
        r2_f[i] * sphericalbesselj_fast(l, p * ri)
    end
end

# # This custom rule makes sure we can use loop vectorisation in the innermost (simpson) loop
# # and that we avoid allocating a Vector{<:Dual} each time hankel( ) is called
# @timing function hankel(r::AbstractVector, r2_f::AbstractVector, l::Integer, p::TT) where {TT <: ForwardDiff.Dual}
#     # Note that
#     #   d H[f] / dp = 4\pi \int_0^∞ r^2 f(r) j_l'(p⋅r)⋅r dr
#     # and that
#     #   j_l'(x) = l / x * j_l(x) - j_{l+1}(x)
# 
#     T  = ForwardDiff.valtype(TT)
#     pv = ForwardDiff.value(p)
# 
#     jl = sphericalbesselj_fast.(l, pv .* r)
#     value = 4T(π) * simpson((i, r) -> r2_f[i] * jl[i], r)
# 
#     if iszero(pv)
#         return TT(value, zero(T) * ForwardDiff.partials(p))
#     end
# 
#     derivative_value = (i, r) -> (r2_f[i] * (l * jl[i] / pv
#                                              - r .* sphericalbesselj_fast(l+1, pv * r)))
#     derivative = 4T(π) * simpson(derivative_value, r)
#     TT(value, derivative * ForwardDiff.partials(p))
# end
