# Explicit GPU kernels for the remapping of data in K-point specific FFTs, using
# KernelAbstractions to remain vendor agnostic. Constructs such as 
# f_real[Gvec_mapping] = f_fourier or f_fourier .= view(f_real, Gvec_mapping)
# are not optimized on the GPU, and performance suffers when large numbers of
# FFT calls take place (tens of thousands or more).

import KernelAbstractions: get_backend, @kernel, @index, @Const

# Mapping kernel for the K-point specific iFFT
@kernel function ifft_remap_kernel!(f_real, @Const(Gvec_mapping), @Const(f_fourier))
    i = @index(Global)
    @inbounds f_real[Gvec_mapping[i]] = f_fourier[i]
end
function ifft_remap!(f_real, Gvec_mapping, f_fourier)
    backend = get_backend(f_real)
    kernel = ifft_remap_kernel!(backend)
    kernel(f_real, Gvec_mapping, f_fourier; ndrange=length(f_fourier))
end

# Mapping kernel for the K-point specific FFT
@kernel function fft_remap_kernel!(@Const(f_real), @Const(Gvec_mapping), f_fourier)
    i = @index(Global)
    @inbounds f_fourier[i] = f_real[Gvec_mapping[i]]
end
function fft_remap!(f_fourier, f_real, Gvec_mapping)
    backend = get_backend(f_fourier)
    kernel = fft_remap_kernel!(backend)
    kernel(f_real, Gvec_mapping, f_fourier; ndrange=length(f_fourier))
end