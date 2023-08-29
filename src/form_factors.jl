import AtomicPotentials.Interpolation: Interpolator

@doc raw"""
Compute the angular part of the form factors
```math
f_{lm,Y}(\mathbf{Q}) = (-i)^l Y_{lm}(\mathbf{Q})
```
where $l$ is the angular momentum, $m$ the magnetic quantum number, and $G$ a vector in
reciprocal space in Cartesian units.
"""
function compute_form_factor_angular(
    Q_cart::Vec3{T}, l::Integer, m::Integer
)::Complex{T} where {T<:Real}
    (-im)^l * ylm_real(l, m, Q_cart)
end
function compute_form_factors_angular(
    Qs_cart::AbstractArray{<:Vec3}, l::Integer, m::Integer
)
    map(Q -> compute_form_factor_angular(Q, l, m), Qs_cart)
end
function compute_form_factors_angular(Qs_cart::AbstractArray{<:Vec3}, l::Integer)
    map(m -> compute_form_factors_angular(Qs_cart, l, m), (-l):(+l))
end
function compute_form_factors_angular(
    Qs_cart::AbstractArray{<:Vec3},
    angular_momenta::Union{AbstractVector,UnitRange,NTuple}
)
    map(l -> compute_form_factors_angular(Qs_cart, l), angular_momenta)
end
function compute_form_factors_angular(basis::PlaneWaveBasis, args...)
    compute_form_factors_angular(G_vectors_cart(basis), args...)
end
function compute_form_factors_angular(basis::PlaneWaveBasis, kpt::Kpoint, args...)
    compute_form_factors_angular(Gplusk_vectors_cart(basis, kpt), args...)
end

@doc raw"""
Compute the radial part of the form factors of an atomic radial function $g(r)$
```math
f_{lm,R}(\mathbf{Q}) = 4\pi \int_{0}^{\infty} r g(r) j_l(|\mathbf{Q}|r) r dr
```
"""
# Lowest level: interface with quantity
function compute_form_factors_radial(
    qs_cart::AbstractArray{<:Real}, qty::AbstractQuantity{RealSpace,Numerical};
    qgrid=qs_cart, quadrature_method=AtomicPotentials.NumericalQuadrature.Trapezoid(),
    interpolation_method=AtomicPotentials.Interpolation.InterpolationsLinear()
)
    qty_fourier = rft(qty, qgrid; quadrature_method)
    itp = evaluate(qty_fourier, interpolation_method)
    map(itp, qs_cart)
end
function compute_form_factors_radial(
    qs_cart::AbstractArray{<:Real}, qty_fourier::AbstractQuantity{FourierSpace,Numerical};
    interpolation_method=AtomicPotentials.Interpolation.InterpolationsLinear(),
    kwargs...
)
    itp = evaluate(qty_fourier, interpolation_method)
    map(itp, qs_cart)
end
function compute_form_factors_radial(
    qs_cart::AbstractArray{<:Real},
    qty;  # ::Union{AbstractQuantity{FourierSpace,Analytical},Interpolator}
    kwargs...
)
    map(qty, qs_cart)
end
function compute_form_factors_radial(
    qs_cart::AbstractArray{<:Real}, qty::AbstractQuantity{RealSpace,Analytical}; kwargs...
)
    compute_form_factors_radial(qs_cart, rft(qty))
end
# Middle level: multiple quantities, Q vector norms
function compute_form_factors_radial(
    qs_cart::AbstractArray{<:Real}, qtys::AbstractVector{<:Union{AbstractQuantity,Interpolator}}; kwargs...
)
    map(qty -> compute_form_factors_radial(qs_cart, qty; kwargs...), qtys)
end
function compute_form_factors_radial(
    qs_cart::AbstractArray{<:Real},
    qtys::AbstractVector{<:AbstractVector{<:Union{AbstractQuantity,Interpolator}}};
    kwargs...
)
    map(qtys_i -> compute_form_factors_radial(qs_cart, qtys_i; kwargs...), qtys)
end
function compute_form_factors_radial(
    qs_cart::AbstractArray{<:Real},
    qtys::AbstractVector{<:AbstractVector{<:AbstractVector{<:Union{AbstractQuantity,Interpolator}}}};
    kwargs...
)
    map(qtys_a -> compute_form_factors_radial(qs_cart, qtys_a; kwargs...), qtys)
end
# Upper middle level: Any number of quantities, Q vectors
function compute_form_factors_radial(
    Qs_cart::AbstractArray{<:Vec3}, qty; kwargs...
)
    compute_form_factors_radial(norm.(Qs_cart), qty; kwargs...)
end
# Higest level: Any number of quantities, basis
function compute_form_factors_radial(
    basis::PlaneWaveBasis, qty; kwargs...
)
    compute_form_factors_radial(
        G_vectors_cart(basis), qty;
        qgrid=basis.atom_qgrid, quadrature_method=basis.atom_rft_quadrature_method,
        interpolation_method=basis.atom_q_interpolation_method
    )
end
function compute_form_factors_radial(
    basis::PlaneWaveBasis, kpt::Kpoint, qty
)
    compute_form_factors_radial(
        Gplusk_vectors_cart(basis, kpt), qty;
        qgrid=basis.atom_qgrid, quadrature_method=basis.atom_rft_quadrature_method,
        interpolation_method=basis.atom_q_interpolation_method
    )
end
