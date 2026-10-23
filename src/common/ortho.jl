@timing function ortho_qr(φk::ArrayType) where {ArrayType <: AbstractArray}
    x = convert(ArrayType, qr(φk).Q)
    if size(x) == size(φk)
        return x
    else
        # Sometimes QR (but funnily not always) CUDA messes up the size here
        return x[:, 1:size(φk, 2)]
    end
end

invsqrt(x) = 1 / sqrt(x)
divided_difference(::typeof(invsqrt), x, y) = -1 / (sqrt(x) * sqrt(y) * (sqrt(x) + sqrt(y)))

@timing function ortho_lowdin(φk::AbstractArray)
    O = Hermitian(φk' * φk)
    # Check for linear dependence: eigenvalues close to zero
    λ = eigvals(Hermitian(ForwardDiff.value.(O)))
    @assert minimum(abs, λ) > eps(real(eltype(λ))) * maximum(abs, λ)
    φk * matrix_function(invsqrt, O)
end
