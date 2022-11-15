# Orthonormalize
@timing function ortho_qr(φk::ArrayType) where {ArrayType <: AbstractArray}
    Q = convert(ArrayType, qr(φk).Q)
    # CUDA bug: after the convert line, when φk is m*n rectangular matrix with m > n,
    # Q is not cropped ie only the first size(φk, 2) columns should be kept
    # See https://github.com/JuliaGPU/CUDA.jl/pull/1662
    Q[:, 1:size(φk, 2)]
end

# CPU specialisation to go a bit faster
@timing ortho_qr(φk::ArrayType) where {ArrayType <: Array} = Array(qr(φk).Q)
