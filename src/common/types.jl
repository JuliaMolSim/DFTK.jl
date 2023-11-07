using StaticArrays
using StaticArrays: setindex

# Deduce the value type (floating-point type for storing plain an static data
# in Model and PlaneWaveBasis) from e.g. an interval or a dual type.
value_type(T) = T

# Frequently-used array types
const Mat3{T} = SMatrix{3, 3, T, 9} where {T}
const Vec3{T} = SVector{3, T} where {T}
const AbstractArray3{T} = AbstractArray{T, 3}
