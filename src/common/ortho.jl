# Orthonormalize and return an array of the same type as the input.
ortho_qr(φk::AbstractArray; array_type = Matrix) = array_type(qr(φk).Q)
