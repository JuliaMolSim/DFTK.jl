# Orthonormalize and convert to an array of the type array_type.
ortho_qr(φk::AbstractArray; array_type = Matrix) = array_type(qr(φk).Q)
