@timing function ortho_qr(φk::ArrayType) where {ArrayType <: AbstractArray}
    x = convert(ArrayType, qr(φk).Q)
    if size(x) == size(φk)
        return x
    else
        # Sometimes QR (but funnily not always) CUDA messes up the size here
        return x[:, 1:size(φk, 2)]
    end
end

@timing function ortho_lowdin(φk::ArrayType) where {ArrayType <: AbstractArray}
    S = φk' * φk
    evals, evecs = eigen(S)
    # Check for linear dependence: eigenvalues close to zero
    tol = maximum(size(φk)) * eps(real(eltype(φk))) * maximum(abs, evals)
    if any(abs.(evals) .< tol)
        error("Input vectors are linearly dependent or nearly so (small eigenvalues detected).")
    end
    ihS = evecs * Diagonal(evals .^ (-0.5)) * evecs'
    x = φk * ihS
    return x
end

