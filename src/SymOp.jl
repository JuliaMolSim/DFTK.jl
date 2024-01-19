# A symmetry operation (SymOp) is a couple (W, w) of a matrix W and a translation w, in
# reduced coordinates, such that for each atom of type A at position a, W a + w is also an
# atom of type A.
# The matrix W is unitary in Cartesian coordinates, but not in reduced coordinates.
# This induces an operator
#   (Uu)(x) = u(W x + w)
# or in Fourier space
#   (Uu)(G) = e^{-i G τ} u(S^-1 G)
# with
#   S = W'
#   τ = -W^-1 w
# (Omitting a 2π factor, all these formulas are valid both in reduced and Cartesian
# coordinates.)

# Time-reversal symmetries are the anti-unitaries
#   (Uu)(x) = conj(u(Wx+w))
# or in Fourier space
#   (Uu)(G) = e^{i G τ} conj(u(-S^-1 G))

# Tolerance to consider two atomic positions as equal (in relative coordinates).
const SYMMETRY_TOLERANCE = convert(Float64, @load_preference("symmetry_tolerance", 1e-5))

# Whether symmetry determination and k-point reduction is checked explicitly in the code
const SYMMETRY_CHECK = true

is_approx_integer(r; atol=SYMMETRY_TOLERANCE) = all(ri -> abs(ri - round(ri)) ≤ atol, r)

struct SymOp{T <: Real}
    # (Uu)(x) = u(W x + w) in real space
    W::Mat3{Int}
    w::Vec3{T}

    # (Uu)(G) = e^{-i G τ} u(S^-1 G) in reciprocal space
    S::Mat3{Int}
    τ::Vec3{T}
end
function SymOp(W, w::AbstractVector{T}) where {T}
    w = mod.(w, 1)
    S = W'
    τ = -W \ w
    SymOp{T}(W, w, S, τ)
end

function Base.convert(::Type{SymOp{T}}, S::SymOp{U}) where {U <: Real, T <: Real}
    SymOp{T}(S.W, T.(S.w), S.S, T.(S.τ))
end

Base.:(==)(op1::SymOp, op2::SymOp) = op1.W == op2.W && op1.w == op2.w
function Base.isapprox(op1::SymOp, op2::SymOp; atol=SYMMETRY_TOLERANCE)
    op1.W == op2.W && is_approx_integer(op1.w - op2.w; atol)
end
Base.one(::Type{SymOp}) = one(SymOp{Bool})  # Not sure about this method
Base.one(::Type{SymOp{T}}) where {T} = SymOp(Mat3{Int}(I), Vec3(zeros(T, 3)))
Base.one(::SymOp{T}) where {T} = one(SymOp{T})
Base.isone(op::SymOp) = isone(op.W) && iszero(op.w)

# group composition and inverse.
function Base.:*(op1::SymOp, op2::SymOp)
    W = op1.W * op2.W
    w = op1.w + op1.W * op2.w
    SymOp(W, w)
end
Base.inv(op::SymOp) = SymOp(inv(op.W), -op.W\op.w)

function check_group(symops::Vector; kwargs...)
    is_approx_in_symops(s1) = any(s -> isapprox(s, s1; kwargs...), symops)
    is_approx_in_symops(one(SymOp)) || error("check_group: no identity element")
    for s in symops
        if !is_approx_in_symops(inv(s))
            error("check_group: symop $s with inverse $(inv(s)) is not in the group")
        end
        for s2 in symops
            if !is_approx_in_symops(s*s2) || !is_approx_in_symops(s2*s)
                error("check_group: product is not stable")
            end
        end
    end
    symops
end
