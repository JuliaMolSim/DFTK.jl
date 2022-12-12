@timing function ortho_qr(φk::ArrayType) where {ArrayType <: AbstractArray}
    x = convert(ArrayType, qr(φk).Q)
    if size(x) == size(φk)
        return x
    else
        # Sometimes QR (but funnily not always) CUDA messes up the size here
        return x[:, 1:size(φk, 2)]
    end
end
