@doc raw"""
Compute structure factors
```math
F_{j,hkl}= \exp \left[ -2πi \mathbf{G} \cdot \mathbf{R}_j \right]
```
where $j$ is the atom index, $\mathbf{G} = \left[ h~k~l \right]$, and $\mathbf{R}_j$ is the
position of atom $j$ in fractional units.
"""
compute_structure_factor(Q::Vec3, R::Vec3) = cis2pi(-dot(Q, R))
function compute_structure_factors(Qs::AbstractArray{<:Vec3}, R::Vec3)
    map(Base.Fix2(compute_structure_factor, R), Qs)
end
function compute_structure_factors(
    Qs::AbstractArray{<:Vec3}, Rs::AbstractVector{<:Vec3}
)
    return map(Base.Fix1(compute_structure_factors, Qs), Rs)
end
function compute_structure_factors(
    Qs::AbstractArray{<:Vec3}, Rs::AbstractVector{<:AbstractVector{<:Vec3}}
)
    return map(Base.Fix1(compute_structure_factors, Qs), Rs)
end
function compute_structure_factors(basis::PlaneWaveBasis, R)
    compute_structure_factors(G_vectors(basis), R)
end
function compute_structure_factors(basis::PlaneWaveBasis, kpt::Kpoint, R)
    compute_structure_factors(Gplusk_vectors(basis, kpt), R)
end

@doc raw"""
Compute the gradient of the structure factors w.r.t. the position `R`.
```math
\mathbf{\nabla F}_{j,hkl} = -2\pi i \left[ h~k~l \right] F_{j,hkl}
```
"""
function compute_structure_factor_gradient(Q::Vec3, R::Vec3{T}) where {T}
    return -2T(π) * im * Q * cis2pi(-dot(Q, R))
end
function compute_structure_factor_gradients(Qs::AbstractArray{<:Vec3}, R::Vec3)
    map(Base.Fix2(compute_structure_factor_gradient, R), Qs)
end
function compute_structure_factor_gradients(
    Qs::AbstractArray{<:Vec3}, Rs::AbstractVector{<:Vec3}
)
    return map(Base.Fix1(compute_structure_factor_gradients, Qs), Rs)
end
function compute_structure_factor_gradients(
    Qs::AbstractArray{<:Vec3}, Rs::AbstractVector{<:AbstractVector{<:Vec3}}
)
    return map(Base.Fix1(compute_structure_factor_gradients, Qs), Rs)
end
function compute_structure_factor_gradients(basis::PlaneWaveBasis, R)
    compute_structure_factor_gradients(G_vectors(basis), R)
end
function compute_structure_factor_gradients(basis::PlaneWaveBasis, kpt::Kpoint, R)
    compute_structure_factor_gradients(Gplusk_vectors(basis, kpt), R)
end
