@views @timing function ortho_qr(basis, kpt, φk::AbstractArray{T}) where {T}
    x = convert(Matrix{T}, qr(to_composite_σG(basis, kpt, φk)).Q)
    # Sometimes QR (but funnily not always) CUDA messes up the size here.
    from_composite_σG(basis, kpt, x)[:, :, 1:size(φk, 3)]
end
