@timing function ortho_qr(φk::ArrayType) where {ArrayType <: AbstractArray}
    x = convert(ArrayType, qr(φk).Q)
    if size(x) == size(φk)
        return x
    else
        # Sometimes QR (but funnily not always) CUDA messes up the size here
        return x[:, 1:size(φk, 2)]
    end
end

@timing function ortho_lowdin(φk::AbstractArray{T}) where {T}
    φk * lowdin_overlap(φk).O_invsqrt
end

"""
Eigendecomposition `O = evecs * Diagonal(evals) * evecs'` of the overlap `O = φk' * φk`
and `O_invsqrt = O^{-1/2}`, such that `φk * O_invsqrt` has orthonormal columns.
"""
function lowdin_overlap(φk::AbstractArray{T}) where {T}
    evals, evecs = eigen(Hermitian(φk' * φk))
    # Check for linear dependence: eigenvalues close to zero
    @assert minimum(abs, evals) > eps(real(T)) * maximum(abs, evals)
    O_invsqrt = evecs * Diagonal(evals .^ (-0.5)) * evecs'
    (; evals, evecs, O_invsqrt)
end

"""
Derivative of `O^{-1/2}` along the Hermitian perturbation `δO`, given the
eigendecomposition `(; evals, evecs)` of `O` (see [`lowdin_overlap`](@ref)).
"""
function lowdin_overlap_derivative(lowdin, δO)
    (; evals, evecs) = lowdin
    # In the eigenbasis of O the derivative is the divided difference
    #   (e_i^{-1/2} - e_j^{-1/2}) / (e_i - e_j) = -1 / (√e_i √e_j (√e_i + √e_j)),
    # written in a form that stays finite for degenerate eigenvalues.
    s = sqrt.(evals)
    K = [-1 / (s[i] * s[j] * (s[i] + s[j])) for i in eachindex(s), j in eachindex(s)]
    evecs * ((evecs' * δO * evecs) .* K) * evecs'
end

