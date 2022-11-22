"""
Abstract supertype for architectures supported by DFTK.
"""
abstract type AbstractArchitecture end

struct CPU <: AbstractArchitecture end

struct GPU{ArrayType <: AbstractArray} <: AbstractArchitecture end

"""
Construct a particular GPU architecture by passing the ArrayType
"""
GPU(::Type{T}) where {T <: AbstractArray} = GPU{T}()

"""
Transfer an array from a device (typically a GPU) to the CPU.
"""
to_cpu(x::AbstractArray) = Array(x)
to_cpu(x::Array) = x

"""
Transfer an array to a particular device (typically a GPU)
"""
to_device(::CPU, x) = to_cpu(x)
to_device(::GPU{ArrayType}, x::AbstractArray) where {ArrayType} = ArrayType(x)
to_device(::GPU{ArrayType}, x::ArrayType)     where {ArrayType} = x
