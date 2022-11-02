# Orthonormalize and convert to an array of the type "array_type".
@timing ortho_qr(φk::AbstractArray; array_type = Matrix) = array_type(qr(φk).Q)

# @timing function ortho_qr(φk::AbstractArray; array_type = Matrix)
#     Q = qr(φk).Q
#     println(typeof(φk))
#     print(size(Array(Q)))
#     res = convert(typeof(φk), Q)
#     print(size(res))
#     res
# end
