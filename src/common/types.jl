using StaticArrays

# Frequently-used array types
const Mat3{T} = SMatrix{3, 3, T, 9} where T
const Vec3{T} = SVector{3, T} where T
const AbstractArray3{T} = AbstractArray{T, 3}

# Represents a symmetry (S,τ)
const SymOp = Tuple{Mat3{Int}, Vec3{Float64}}
identity_symop() = (Mat3{Int}(I), Vec3(zeros(3)))
