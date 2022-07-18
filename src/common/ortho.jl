# Orthonormalize
ortho_qr(φk::AbstractArray) = Matrix(qr(φk).Q) #LinearAlgebra.QRCompactWYQ -> Matrix
ortho_qr(φk::CuArray) = CuArray(qr(φk).Q) #CUDA.CUSOLVER.CuQRPackedQ -> CuArray
