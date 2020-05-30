using StaticArrays

# Frequently-used static types
const Mat3{T} = SMatrix{3, 3, T, 9} where T
const Vec3{T} = SVector{3, T} where T
const AbstractArray3{T} = AbstractArray{T, 3}
