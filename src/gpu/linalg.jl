### GPU-specific implementations of DFTK's column-wise linear algebra helpers.
# The massive parallelism of the GPU can only be fully exploited when
# operating on whole arrays. For performance reasons, one should avoid
# explicitly looping over columns or elements. This approach is not
# necessarily the most performant on CPU, as the allocation of large
# temporary arrays hurts cache locality. It is also harder to read.
# (The analogous helpers used *inside* the eigensolver live in LOBPCGEigensolver.jl.)

using LinearAlgebra
using GPUArraysCore

function columnwise_dots(A::AbstractGPUArray{T}, B::AbstractGPUArray{T}) where {T}
    vec(sum(conj(A) .* B; dims=1))
end

function columnwise_dots(A::AbstractGPUArray{T}, M, B::AbstractGPUArray{T}) where {T}
    vec(sum(conj(A) .* (M * B); dims=1))
end

function columnwise_dots(A::AbstractGPUArray{T}, D::Diagonal, B::AbstractGPUArray{T}) where {T}
    vec(sum(conj(A) .* (D.diag .* B); dims=1))
end

function ldiv!(Y::AbstractGPUArray{T}, P::PreconditionerTPA, R::AbstractGPUArray{T}) where {T}
    if P.mean_kin === nothing
        ldiv!(Y, Diagonal(P.kin .+ P.default_shift), R)
    else
        Y .= (P.mean_kin' ./ (P.mean_kin' .+ P.kin)) .* R
    end
    Y
end

function mul!(Y::AbstractGPUArray{T}, P::PreconditionerTPA, R::AbstractGPUArray{T}) where {T}
    if P.mean_kin === nothing
        mul!(Y, Diagonal(P.kin .+ P.default_shift), R)
    else
        Y .= ((P.mean_kin' .+ P.kin) ./ P.mean_kin') .* R
    end
    Y
end

# GPU-aware `svd` for Hermitian matrices backed by GPU arrays.
# TODO: upstream this method
function LinearAlgebra.svd(A::Union{Hermitian{T, <:AbstractGPUArray{T}},
                                    Hermitian{Complex{T}, <:AbstractGPUArray{Complex{T}}}};
                           full::Bool=false) where {T<:Real}
    E = eigen(A)
    λ, U = E.values, E.vectors

    # The singular-value ordering is a tiny CPU computation, but the singular
    # values themselves are kept on the device (like the backend svd methods do),
    # so that downstream GPU-side assembly (e.g. in `pinv`) stays device-consistent.
    σ = to_cpu(abs.(λ))
    p = sortperm(σ; rev=true)
    arch = architecture(parent(A))
    S = to_device(arch, σ[p])
    Up = U[:, p]
    signs = to_device(arch, sign.(λ[p]))  # sign of the eigenvalues (not of σ!)

    # Vt[i, j] = signᵢ * conj(Up[j, i]), i.e. Vt = (Up * Diagonal(signs))†
    Vt = adjoint(Up) .* permutedims(signs)
    SVD(Up, S, Vt)
end
