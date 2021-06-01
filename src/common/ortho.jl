# Orthonormalises a set of planewave at a given k-point
ortho_qr(φk) = Matrix(qr(φk).Q)
