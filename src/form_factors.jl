@doc raw"""
Compute the angular part of the form factors
```math
f_{lm,Y}(\mathbf{q}) = -i^l Y_{lm}(\mathbf{G})
```
where $l$ is the angular momentum, $m$ the magnetic quantum number, and $G$ a vector in
reciprocal space in Cartesian units.
"""
function compute_form_factor_angular(G_cart::Vec3{T}, l::Integer, m::Integer) where {T}
    return (-im)^l * ylm_real(l, m, G_cart)
end
function compute_form_factors_angular(
    Gs_cart::AbstractArray{Vec3{T}}, l::Integer, m::Integer
) where {T}
    map(G -> compute_form_factor_angular(G, l, m), Gs_cart)
end
function compute_form_factors_angular(
    Gs_cart::AbstractArray{Vec3{T}}, l::Integer
) where {T}
    map(m -> compute_form_factors_angular(Gs_cart, l, m), (-l):(+l))
end
