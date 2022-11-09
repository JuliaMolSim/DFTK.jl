"""
Abstract supertype for architectures supported by DFTK.
"""
abstract type AbstractArchitecture end

struct CPU <: AbstractArchitecture end

get_array_type(::CPU) = Array

"""
Generic, hardware independent architecture for DFTK.
"""
abstract type GPU <: AbstractArchitecture end
