@views blochwave_as_tensor(ψk::AbstractMatrix, n_components) = reshape(ψk, n_components, :, size(ψk, 2))
@views blochwave_as_matrix(ψk::AbstractArray3) = reshape(ψk, :, size(ψk, 3))
# reduce along component direction
@views blochwave_as_matorvec(ψk::AbstractArray3) = reshape(ψk, :, size(ψk, 3))
@views blochwave_as_matorvec(ψk::AbstractMatrix) = reshape(ψk, size(ψk, 2))
# Works for BlochWaves & Vector(AbstractArray3).
@views blochwaves_as_matrices(ψ) = @views [reshape(ψk, :, size(ψk, 3)) for ψk in ψ]
to_composite_σG() = nothing
from_composite_σG() = nothing
