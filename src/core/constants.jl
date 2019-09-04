using StaticArrays

# Frequently-used static types
Mat3{T} = SMatrix{3, 3, T, 9} where T
Vec3{T} = SVector{3, T} where T

"""The default search location for Pseudopotential data files"""
DFTK_DATADIR = joinpath(dirname(pathof(DFTK)), "..", "data")

# Special macro to flag expensive assertions, which can be
# disabled by setting these to nothing
macro assert_expensive(expr) :(@assert $expr) end
macro assert_expensive(expr, text) :(@assert($expr, $text)) end
# macro assert_expensive(expr) nothing end
# macro assert_expensive(expr, text) nothing end
