"""
Specialised architecture for NVIDIA CUDA GPUs.
"""
struct CUDAGPU <: GPU end
GPU(::Type{CUDA.CuArray}) = CUDAGPU()

"""
Transfer an array from a device (typically the CPU) to the NVIDIA CUDA GPU.
"""
to_device(::CUDAGPU, x::AbstractArray) = CUDA.CuArray(x)
to_device(::CUDAGPU, x::CUDA.CuArray) = x
