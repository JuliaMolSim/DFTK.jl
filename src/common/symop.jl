# Represents a symmetry (S,τ)
struct SymOp
    S::Mat3{Int}
    τ::Vec3{Float64}
    function SymOp(S, τ)
        τ = τ .- floor.(τ)
        @assert all(0 .≤ τ .< 1)
        new(S, τ)
    end
    SymOp(Sτ::Tuple) = SymOp(Sτ...) # compatibility with old stuff, doesn't hurt
end

Base.:(==)(op1::SymOp, op2::SymOp) = op1.S == op2.S && isapprox(op1.τ, op2.τ; atol=1e-8)
Base.one(::Type{SymOp}) = SymOp(Mat3{Int}(I), Vec3(zeros(3)))

# group composition and inverse.
# Derived either from the formulas for the composition/inverse of Stilde/τtilde
# then passing to reciprocal, or directly from the symmetry operation in reciprocal space
function Base.:*(op1, op2)
    S = op1.S * op2.S
    τ = op1.τ + op1.S' \ op2.τ
    SymOp(S, τ)
end
Base.inv(op) = SymOp(inv(op.S), -op.S'*op.τ)

function check_group(symops::Vector)
    for s in symops
        inv(s) ∈ symops || error("check_group: symop $s with inverse $(inv(s)) is not in the group")
        for s2 in symops
            (s*s2 ∈ symops && s2*s ∈ symops) || error("check_group: product is not stable")
        end
    end
    symops
end
