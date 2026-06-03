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

function is_approx_integer(r; atol=100eps(eltype(r)))
    # Note: This default on atol is deliberately chosen quite tight. In many applications
    # of this function one probably wants to use SYMMETRY_TOLERANCE as atol.
    all(ri -> abs(ri - round(ri)) ≤ atol, r)
end

struct SymOp{T <: Real}
    # (Uu)(x) = u(W x + w) in real space           (θ = +1, unitary)
    # (Uu)(x) = conj(u(W x + w)) in real space     (θ = -1, antiunitary / conjugation)
    # TODO: for noncollinear spin, this should carry a SU(2) matrix acting on the spinor
    W::Mat3{Int}
    w::Vec3{T}

    # (Uu)(G) = e^{-i G τ} u(S^-1 G) in reciprocal space           (θ = +1)
    # (Uu)(G) = e^{+i G τ} conj(u(-S^-1 G)) in reciprocal space    (θ = -1)
    S::Mat3{Int}
    τ::Vec3{T}

    # θ ∈ {+1, -1}: +1 for unitary, -1 for antiunitary (conjugation, maps k → -k)
    θ::Int
end
function SymOp(W, w::AbstractVector{T}; θ=1) where {T}
    @assert θ ∈ (1, -1)
    w = mod.(w, 1)
    S = W'
    τ = -W \ w
    SymOp{T}(W, w, S, τ, Int(θ))
end

function Base.convert(::Type{SymOp{T}}, op::SymOp{U}) where {U <: Real, T <: Real}
    SymOp{T}(op.W, T.(op.w), op.S, T.(op.τ), op.θ)
end

Base.:(==)(op1::SymOp, op2::SymOp) = op1.W == op2.W && op1.w == op2.w && op1.θ == op2.θ
function Base.isapprox(op1::SymOp, op2::SymOp; atol=SYMMETRY_TOLERANCE)
    op1.W == op2.W && op1.θ == op2.θ && is_approx_integer(op1.w - op2.w; atol)
end
Base.one(::Type{SymOp}) = one(SymOp{Bool})  # Not sure about this method
Base.one(::Type{SymOp{T}}) where {T} = SymOp(Mat3{Int}(I), Vec3(zeros(T, 3)))
Base.one(::SymOp{T}) where {T} = one(SymOp{T})
Base.isone(op::SymOp) = isone(op.W) && iszero(op.w) && op.θ == 1

# group composition: spatial parts compose normally, θ multiplies
function Base.:*(op1::SymOp, op2::SymOp)
    W = op1.W * op2.W
    w = op1.w + op1.W * op2.w
    SymOp(W, w; θ=op1.θ * op2.θ)
end
# inverse: (W^{-1}, -W^{-1}w, θ) — θ unchanged, antiunitary inverse is antiunitary
Base.inv(op::SymOp) = SymOp(inv(op.W), -op.W\op.w; θ=op.θ)

# k → θ·S·k: the BZ action of a symop (maps k-points in reduced coordinates)
transform_kpoint_coordinate(op::SymOp, k) = op.θ * op.S * k

# Conjugate x when the symop is antiunitary (θ=-1), identity otherwise
maybe_conjugate(op::SymOp, x) = op.θ == 1 ? x : conj(x)

is_approx_in(symop, group; kwargs...) = any(s -> isapprox(s, symop; kwargs...), group)
function check_group(symops::Vector; kwargs...)
    is_approx_in_symops(s1) = is_approx_in(s1, symops; kwargs...)
    is_approx_in_symops(one(SymOp)) || error("check_group: no identity element")
    for s in symops
        if !is_approx_in_symops(inv(s))
            error("check_group: symop $s with inverse $(inv(s)) is not in the group")
        end
        for s2 in symops
            if !is_approx_in_symops(s*s2)
                error("check_group: product is not stable: $(s*s2) is not in the group")
            end
        end
    end
    symops
end

function complete_symop_group(symops; maxiter=10, kwargs...)
    completed_group = Vector(symops)

    function add_to_group(to_add, s1)
        if !is_approx_in(s1, completed_group; kwargs...) && !is_approx_in(s1, to_add; kwargs...)
            push!(to_add, s1)
        end
    end

    for it = 1:maxiter
        if it == maxiter
            error("Could not complete group in $maxiter iterations")
        end
        to_add = []
        # Identity always needs to be there!
        add_to_group(to_add, one(SymOp))
        for s in completed_group
            add_to_group(to_add, inv(s))
            for t in completed_group
                add_to_group(to_add, s*t)
            end
        end
        if isempty(to_add)
            return completed_group
        end
        append!(completed_group, to_add)
    end
    DFTK.check_group(completed_group) # returns the completed group
end
