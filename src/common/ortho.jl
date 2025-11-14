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
    evals, evecs = eigen(Hermitian(φk' * φk))
    # Check for linear dependence: eigenvalues close to zero
    @assert minimum(abs, evals) > eps(real(T)) * maximum(abs, evals) 
    ihS = evecs * Diagonal(evals .^ (-0.5)) * evecs'
    φk * ihS
end

