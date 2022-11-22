# Orthonormalize
@timing ortho_qr(φk::ArrayType) where {ArrayType <: AbstractArray} = convert(ArrayType, qr(φk).Q)
