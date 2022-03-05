# A symmetry operation (SymOp) is a couple (W, w) of a
# unitary (in cartesian coordinates, but not in reduced coordinates)
# matrix W and a translation w such that, for each atom of
# type A at position a, W a + w is also an atom of type A.
# This induces an operator
# (Uu)(x) = u(W x + w)
# or in Fourier space
# (Uu)(G) = e^{-i G τ} u(S^-1 G)
# with
# S = W'
# τ = -W^-1 w
# (all these formulas are valid both in reduced and cartesian coordinates)

# Time-reversal symmetries are the anti-unitaries
# (Uu)(x) = (θu)(Wx+w)
# or in Fourier space
# (Uu)(G) = e^{-i G τ} (θu)(S^-1 G), with the modified S = -W', τ = W^-1 w
# where θ is the conjugation operator

# Tolerance to consider two atomic positions as equal (in relative coordinates)
const SYMMETRY_TOLERANCE = 1e-5

struct SymOp{T <: Real}
    # (Uu)(x) = (θu)(W x + w) in real space
    W::Mat3{Int}
    w::Vec3{T}
    θ::Bool # true for TRS, false for no TRS

    # (Uu)(G) = e^{-i G τ} (θu)(S^-1 G) in reciprocal space
    S::Mat3{Int}
    τ::Vec3{T}

    function SymOp(W, w, θ=false)
        w = mod.(w, 1)
        S = W'
        τ = -W \w
        if θ
            S = -S
            τ = -τ
        end
        new{eltype(τ)}(W, w, θ, S, τ)
    end
end

Base.:(==)(op1::SymOp, op2::SymOp) = op1.W == op2.W && op1.w == op2.w && op1.θ == op2.θ
function Base.isapprox(op1::SymOp, op2::SymOp; atol=SYMMETRY_TOLERANCE)
    is_approx_integer(r) = all(ri -> abs(ri - round(ri)) ≤ atol, r)
    op1.W == op2.W && is_approx_integer(op1.w - op2.w) && op1.θ == op2.θ
end
Base.one(::Type{SymOp}) = SymOp(Mat3{Int}(I), Vec3(zeros(Bool, 3)))
Base.one(::SymOp) = one(SymOp)

# group composition and inverse.
function Base.:*(op1, op2)
    W = op1.W * op2.W
    w = op1.w + op1.W * op2.w
    SymOp(W, w, xor(op1.θ, op2.θ))
end
Base.inv(op) = SymOp(inv(op.W), -op.W\op.w, op.θ)

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
