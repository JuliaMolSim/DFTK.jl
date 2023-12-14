struct BlochWaves{T, Tψ, Basis <: AbstractBasis{T}}
    basis::Basis
    data::Vector{VT} where {VT <: AbstractArray3{Tψ}}
    n_components::Int
end
function BlochWaves(basis::PlaneWaveBasis, ψ::Vector{VT}) where {VT <: AbstractArray3}
    BlochWaves(basis, ψ, basis.model.n_components)
end
function BlochWaves(basis::PlaneWaveBasis, ψ::Vector{VT}) where {VT <: AbstractMatrix}
    n_components = basis.model.n_components
    BlochWaves(basis, [reshape(ψk, n_components, :, size(ψk, 2)) for ψk in ψ], n_components)
end
BlochWaves(basis) = BlochWaves(basis, [Array{eltype(basis), 3}(undef, 0, 0, 0)], 0)

# Helpers function to directly have access to the `data` field.
Base.getindex(ψ::BlochWaves, indices...) = ψ.data[indices...]
Base.isnothing(ψ::BlochWaves) = iszero(ψ.data)
Base.iterate(ψ::BlochWaves, args...) = Base.iterate(ψ.data, args...)
Base.length(ψ::BlochWaves) = length(ψ.data)
Base.similar(ψ::BlochWaves, args...) = similar(ψ.data, args...)
Base.size(ψ::BlochWaves, args...) = Base.size(ψ.data, args...)

# T@D@: Do not change flatten the size of array by default (1:1)
# T@D@: replace with iterations with eachslice(ψk; dims=1)

@doc raw"""
    view_component(ψk::AbstractArray3, σ)

View the ``σ``-th component(s) of the wave function `ψk`.
It returns a 2D matrix if `σ` is an integer or a 3D array if it is a list.
"""
@views view_component(ψk::AbstractArray3, σ) = ψk[σ, :, :]
# Apply the previous function for each k-point.
@views view_component(ψ::BlochWaves, σ) = [view_component(ψk, σ) for ψk in ψ]
"""
    denest(ψ::BlochWaves; σ)

Returns the arrays containing the data from the `BlochWaves` structure `ψ`.
If `σ` is given, we can ask for only some components to be extracted.
"""
@views denest(ψ::BlochWaves; σ=1:ψ.basis.model.n_components) = [view_component(ψk, σ) for ψk in ψ]
@views denest(basis, ψ::Vector; σ=1:basis.model.n_components) = [view_component(ψk, σ) for ψk in ψ]
# Wrapper around the BlochWaves creator to have an inverse to the `denest` function.
nest(basis, ψ::Vector{A}) where {A <: AbstractArray} = BlochWaves(basis, ψ)

eachcomponent(ψk::AbstractArray3) = eachslice(ψk; dims=1)
eachband(ψk::AbstractArray3)      = eachslice(ψk; dims=3)

@views blochwave_as_tensor(ψk::AbstractMatrix, n_components) = reshape(ψk, n_components, :, size(ψk, 2))
@views blochwave_as_matrix(ψk::AbstractArray3) = reshape(ψk, :, size(ψk, 3))
# reduce along component direction
@views blochwave_as_matorvec(ψk::AbstractArray3) = reshape(ψk, :, size(ψk, 3))
@views blochwave_as_matorvec(ψk::AbstractMatrix) = reshape(ψk, size(ψk, 2))
# Works for BlochWaves & Vector(AbstractArray3).
@views blochwaves_as_matrices(ψ) = @views [reshape(ψk, :, size(ψk, 3)) for ψk in ψ]
