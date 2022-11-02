# Orthonormalize
@timing function ortho_qr(φk::AbstractArray)
    Q = qr(φk).Q
    Q = convert(typeof(φk), Q)
    # CUDA bug: after the convert line, when φk is m*n rectangular matrix with m > n,
    # Q is not cropped ie only the first size(φk, 2) columns should be kept
    Q[:, 1:size(φk, 2)]
end
