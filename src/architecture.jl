"""
Abstract supertype for architectures supported by DFTK.
"""
abstract type AbstractArchitecture end

struct CPU <: AbstractArchitecture end

"""
Transfer an array from a device (typically a GPU) to the CPU.
"""
to_cpu(x::AbstractArray) = Array(x)
to_cpu(x::Array) = x

to_device(::CPU, x) = to_cpu(x)


"""
Generic, hardware independent architecture for DFTK.
"""
abstract type GPU <: AbstractArchitecture end
