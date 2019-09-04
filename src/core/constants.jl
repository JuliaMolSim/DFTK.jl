using StaticArrays

# Frequently-used static types
Mat3{T} = SMatrix{3, 3, T, 9} where T
Vec3{T} = SVector{3, T} where T

"""The default search location for Pseudopotential data files"""
DFTK_DATADIR = joinpath(dirname(pathof(DFTK)), "..", "data")
