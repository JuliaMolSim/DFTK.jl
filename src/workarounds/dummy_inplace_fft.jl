# A dummy wrapper around an out-of-place FFT plan to make it appear in-place
# This is needed for some FFT implementations, which do not have in-place plans
struct DummyInplace{opFFT}
    fft::opFFT
end

import Base: *, length
Base.:*(p::DummyInplace, X) = copy!(X, p.fft * X)
length(p::DummyInplace) = length(p.fft)
