# Helpers functions converting to and fro implicit composite definition in [σG] and explicit
# definition in [σ,G].
# The `basis` and `kpoint` arguments are not-necessary, but we keep them to match the
# signatures of most DFTK functions.

# ψk[σ, G, n] → ψk[σG, n]
# `prod()` because of Sternheimer solver where some quantities might have 0-ending sizes.
to_composite_σG(::PlaneWaveBasis, ::Kpoint, ψk::AbstractArray3) =
    reshape(ψk, prod(size(ψk)[1:2]), :)
# ψk[σ, G] → ψk[σG]
to_composite_σG(::PlaneWaveBasis, ::Kpoint, ψk::AbstractMatrix) = reshape(ψk, :)
"""
Converts each wave-function from a tensor `ψ[ik][σ, G, n]` to a matrix `ψ[ik][σG, n]`.
Useful for inner-working of DFTK that relies on simple matrix-vector operations.
"""
to_composite_σG(basis, ψ::AbstractVector) = to_composite_σG.(basis, basis.kpoints, ψ)

# ψk[σG, n] → ψk[σ, G, n]
from_composite_σG(basis::PlaneWaveBasis, ::Kpoint, ψk::AbstractMatrix) =
    reshape(ψk, basis.model.n_components, :, size(ψk, 2))
# ψk[σG] → ψk[σ, G]
from_composite_σG(basis::PlaneWaveBasis, ::Kpoint, ψk::AbstractVector) =
    reshape(ψk, basis.model.n_components, :)

# For LOBPCG compability.
# ψk[σG] → ψk[σ, G, 1:1]
unfold_from_composite_σG(basis::PlaneWaveBasis, ::Kpoint, ψk::AbstractVector) =
    reshape(ψk, basis.model.n_components, :, 1)
# ψk[1:1G, n] → ψk[1:1, G, n]
unfold_from_composite_σG(basis::PlaneWaveBasis, ::Kpoint, ψk::AbstractMatrix) =
    reshape(ψk, basis.model.n_components, :, size(ψk, 2))
"""
Reciprocal to [`to_composite_σG`](@ref).
"""
from_composite_σG(basis::PlaneWaveBasis, ψ::AbstractVector) =
    from_composite_σG.(basis, basis.kpoints, ψ)

@timing function enforce_phase!(basis::PlaneWaveBasis, kpt::Kpoint, ψk)
    map(eachcol(to_composite_σG(basis, kpt, ψk))) do ψkn
        ϕ = angle(ψkn[end])
        if abs(ϕ) > sqrt(eps(ϕ))
            ε = sign(real(ψkn[end] / cis(ϕ)))
            ψkn ./= ε*cis(ϕ)
        end
    end
    ψk
end
"""Ensure that the wave function is real."""
enforce_phase!(basis, ψ::AbstractVector) = enforce_phase!.(basis, basis.kpoints, ψ)
