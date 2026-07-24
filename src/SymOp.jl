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
    # (Uu)(x) = u(W x + w) in real space
    W::Mat3{Int}
    w::Vec3{T}

    # (Uu)(G) = e^{-i G τ} u(S^-1 G) in reciprocal space
    S::Mat3{Int}
    τ::Vec3{T}
    invS::Mat3{Int}  # = inv(S), integer since det S = 1
end
function SymOp(W, w::AbstractVector{T}) where {T}
    w = mod.(w, 1)
    S = W'
    τ = -W \ w
    SymOp{T}(W, w, S, τ, Mat3{Int}(inv(S)))
end

function Base.convert(::Type{SymOp{T}}, S::SymOp{U}) where {U <: Real, T <: Real}
    SymOp{T}(S.W, T.(S.w), S.S, T.(S.τ), S.invS)
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

# A star is the orbit {G = S·G₀ : S in the group} of a representative G-vector G₀ on the FFT
# grid. The symmetrized density is constant over a star up to a phase, ρ_sym(S·G₀) =
# e^{-i (S·G₀)·τ_S} ρ_sym(G₀), so `accumulate_over_symmetries!` gathers ρ_sym(G₀) once and
# scatters it across the orbit. (Defined here, before PlaneWaveBasis, as it is a basis field.)
struct SymmetryStar{T}
    # gather: ρstar = Σ_isym θ·ρ[iG] with (iG, θ) = sources[isym]
    sources::Vector{Pair{Int, Complex{T}}}
    # scatter: ρsym[iG] = θ·ρstar for (iG, θ) in members
    members::Vector{Pair{Int, Complex{T}}}
    # G₀ is not stored: it is the sources/members entry of the trivial symmetry.
end

# Discover the stars by scanning the grid: each grid point not already in a star seeds a new one
# as its representative G₀, and its whole orbit is marked `visited` -- so the scan produces one
# star per orbit, not one per grid point.
function compute_symmetry_stars(fft_size, symmetries::AbstractVector{<:SymOp{T}}) where {T}
    Gs         = G_vectors(fft_size)
    linear_ind = LinearIndices(fft_size)
    visited    = falses(length(Gs))
    stars      = SymmetryStar{T}[]
    for lin in eachindex(Gs)
        visited[lin] && continue
        G = Gs[lin]
        sources = Pair{Int, Complex{T}}[]
        for symop in symmetries
            idx = index_G_vectors(fft_size, symop.invS * G)
            isnothing(idx) || push!(sources, linear_ind[idx] => cis2pi(-T(dot(G, symop.τ))))
        end
        members = Pair{Int, Complex{T}}[]
        for symop in symmetries
            SG  = symop.S * G
            idx = index_G_vectors(fft_size, SG)
            (isnothing(idx) || visited[linear_ind[idx]]) && continue
            visited[linear_ind[idx]] = true
            push!(members, linear_ind[idx] => cis2pi(-T(dot(SG, symop.τ))))
        end
        push!(stars, SymmetryStar(sources, members))
    end
    stars
end
