@doc raw"""
Compute structure factors
```math
F_{j,hkl}= \exp \left[ -2πi \mathbf{G} \cdot \mathbf{R}_j \right]
```
where $j$ is the atom index, $\mathbf{G} = \left[ h~k~l \right]$, and $\mathbf{R}_j$ is the
position of atom $j$ in fractional units.
"""
compute_structure_factor(G::Vec3, position::Vec3) = cis2pi(-dot(G, position))
function compute_structure_factors(Gs::AbstractArray, position::Vec3{T}) where {T}
    map(G -> cis2pi(-dot(G, position)), Gs)
end
function compute_structure_factors(
    Gs::AbstractArray, positions::AbstractVector{Vec3{T}}
) where {T}
    return map(position -> compute_structure_factors(Gs, position), positions)
end
function compute_structure_factors(
    basis::PlaneWaveBasis{T}, kpt::Kpoint{T}, positions
) where {T}
    Gpks = Gplusk_vectors(basis, kpt)
    map(positions_a -> compute_structure_factors(Gpks, positions_a), positions)
end
function compute_structure_factors(basis::PlaneWaveBasis{T}, positions) where {T}
    Gs = G_vectors(basis)
    map(positions_a -> compute_structure_factors(Gs, positions_a), positions)
end

@doc raw"""
Compute the gradient of the structure factors w.r.t. the position `position`.
```math
\mathbf{\nabla F}_{j,hkl} = -2\pi i \left[ h~k~l \right] F_{j,hkl}
```
"""
function compute_structure_factor_gradient(G::Vec3, position::Vec3{T}) where {T}
    return -2T(π) * im * G * cis2pi(-dot(G, position))
end
function compute_structure_factor_gradients(Gs::AbstractArray, position::Vec3{T}) where {T}
    return map(Gs) do G
        compute_structure_factor_gradient(G, position)
    end
end
function compute_structure_factor_gradients(
    Gs::AbstractArray, positions::AbstractVector{Vec3{T}}
) where {T}
    return map(position -> compute_structure_factor_gradients(Gs, position), positions)
end
function compute_structure_factor_gradients(
    basis::PlaneWaveBasis{T}, kpt::Kpoint{T}, positions
) where {T}
    Gpks = Gplusk_vectors(basis, kpt)
    map(positions_a -> compute_structure_factor_gradients(Gpks, positions_a), positions)
end
function compute_structure_factor_gradients(basis::PlaneWaveBasis{T}, positions) where {T}
    Gs = G_vectors(basis)
    map(positions_a -> compute_structure_factor_gradients(Gs, positions_a), positions)
end
