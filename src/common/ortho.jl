@timing function ortho_qr(φk::AbstractArray{T}) where {T}
    x = convert(Matrix{T}, qr(blochwave_as_matrix(φk)).Q)
    blochwave_as_tensor(x, size(φk, 1))[:, :, 1:size(φk, 3)]
end
