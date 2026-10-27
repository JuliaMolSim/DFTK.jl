@timing function ortho_qr(φk::ArrayType) where {ArrayType <: AbstractArray}
    fact = qr(φk)
    # The phase of the Q vectors in a QR factorization is arbitrary.
    # We fix the phase of Q (i.e. enforcing positive real R diagonal)
    # to ensure that ortho_qr is always first-order accurate
    # which phase shifts could otherwise break.
    x = convert(ArrayType, fact.Q) * Diagonal(sign.(diag(fact.R)))
    if size(x) == size(φk)
        return x
    else
        # Sometimes QR (but funnily not always) CUDA messes up the size here
        return x[:, 1:size(φk, 2)]
    end
end

@timing function ortho_lowdin(φk::AbstractArray{T}) where {T}
    evals, evecs = eigen(Hermitian(φk' * φk))
    # Check for linear dependence: eigenvalues close to zero
    @assert minimum(abs, evals) > eps(real(T)) * maximum(abs, evals) 
    ihS = evecs * Diagonal(evals .^ (-0.5)) * evecs'
    φk * ihS
end

