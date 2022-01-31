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
