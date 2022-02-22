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

# Tolerance to consider two atomic positions as equal (in relative coordinates)
const SYMMETRY_TOLERANCE = 1e-5

# Represents a symmetry (S,τ)
struct SymOp
    S::Mat3{Int}
    τ::Vec3{Float64}  # floating-point type fixed by spglib
    function SymOp(S, τ)
        τ = mod.(τ, 1)
        new(S, τ)
    end
    # compatibility with old stuff, will be removed at some point but doesn't hurt for now
    SymOp(Sτ::Tuple) = SymOp(Sτ...)
end

function get_Ww(op::SymOp)
    W = op.S'  # S = W'
    w = -W*op.τ  # τ = -W^-1 w
    (W, w)
end

Base.:(==)(op1::SymOp, op2::SymOp) = op1.S == op2.S && op1.τ == op2.τ
function Base.isapprox(op1::SymOp, op2::SymOp; atol=SYMMETRY_TOLERANCE)
    is_approx_integer(r) = all(ri -> abs(ri - round(ri)) ≤ atol, r)
    op1.S == op2.S && isapprox(op1.τ - op2.τ; atol)
end
Base.one(::Type{SymOp}) = SymOp(Mat3{Int}(I), Vec3(zeros(3)))
Base.one(::SymOp) = one(SymOp)

# group composition and inverse.
# Derived either from the formulas for the composition/inverse of W/w
# then passing to reciprocal, or directly from the symmetry operation in reciprocal space
function Base.:*(op1, op2)
    S = op1.S * op2.S
    τ = op1.τ + op1.S' \ op2.τ
    SymOp(S, τ)
end
Base.inv(op) = SymOp(inv(op.S), -op.S'*op.τ)

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
