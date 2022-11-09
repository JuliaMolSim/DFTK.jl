"""
Specialised architecture for NVIDIA CUDA GPUs.
"""
struct CUDAGPU <: GPU end
GPU(::Type{CUDA.CuArray}) = CUDAGPU()

get_array_type(::CUDAGPU) = CUDA.CuArray
