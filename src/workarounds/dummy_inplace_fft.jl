# A dummy wrapper around an out-of-place FFT plan to make it appear in-place
# This is needed for some generic FFT implementations, which do not have in-place plans
struct DummyInplace{opFFT}
    fft::opFFT
end
LinearAlgebra.mul!(Y, p::DummyInplace, X) = (Y .= mul!(similar(X), p.fft, X))
LinearAlgebra.ldiv!(Y, p::DummyInplace, X) = (Y .= ldiv!(similar(X), p.fft, X))

import Base: *, \, length
*(p::DummyInplace, X) = p.fft * X
\(p::DummyInplace, X) = p.fft \ X
length(p::DummyInplace) = length(p.fft)
