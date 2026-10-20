# Simple BLAS-capable matmul for Dual-valued matrices.
# We extract the value and each partial into plain arrays of BlasFloat,
# call ordinary matmul on those, and rebuild the Dual result.
# See: https://github.com/JuliaDiff/ForwardDiff.jl/issues/854

# Helper type for Dual/Complex{Dual}
const DualEltype{T,V,N} = Union{Dual{T,V,N}, Complex{Dual{T,V,N}}}

# Helper type for strict scalars
const ScalarEltype = Union{AbstractFloat, Complex{<:AbstractFloat}}

function dual_matrix_value_and_partials(A::AbstractMatrix{<:DualEltype{T,V,N}}, npartials::Int) where {T,V,N}
    (ForwardDiff.value.(A), ntuple(p -> ForwardDiff.partials.(A, p), npartials))
end
function dual_matrix_value_and_partials(A::AbstractMatrix, npartials::Int)
    (A, ntuple(_ -> nothing, npartials))
end

# Performance workaround for GEMM of matrices with Dual/Complex{Dual} element types, ensuring
# that the matrix multiplication is dispatched to BLAS instead of the generic Julia implementation.
# This is a generi 5-argument multiplication, which is then called by all overloaded methods.
function _dual_mul!(C::AbstractMatrix{<:DualEltype{T,V,N}}, A, B, α::Number, β::Number) where {T,V,N}
    S = eltype(C)
    A_val, A_parts = dual_matrix_value_and_partials(A, N)
    B_val, B_parts = dual_matrix_value_and_partials(B, N)

    # Value type underlying the Dual element (V for Dual, Complex{V} for Complex{Dual}).
    VT = ForwardDiff.valtype(S)

    # Value part: C_val = α * A_val * B_val + β * C_val
    C_val = similar(C, VT)
    if iszero(β)
        fill!(C_val, zero(VT))
    else
        C_val .= ForwardDiff.value.(C)
    end
    mul!(C_val, A_val, B_val, α, β)

    # Partial parts, i.e. cross products of A and B values and partials
    C_parts = ntuple(N) do p
        Cp = similar(C, VT)
        if iszero(β)
            fill!(Cp, zero(VT))
        else
            Cp .= ForwardDiff.partials.(C, p)
        end
        Ap = A_parts[p]
        Bp = B_parts[p]
        if isnothing(Ap) && isnothing(Bp)
            # Cp already contains β*∂C_p
        elseif isnothing(Ap)
            mul!(Cp, A_val, Bp, α, β)
        elseif isnothing(Bp)
            mul!(Cp, Ap, B_val, α, β)
        else
            mul!(Cp, Ap, B_val, α, β)
            mul!(Cp, A_val, Bp, α, true)
        end
        Cp
    end

    # Reconstruct C as matrix of Dual/Complex{Dual} from value and partial arrays.
    if S <: Dual
        map!((val, parts...) -> Dual{T,V,N}(val, ForwardDiff.Partials{N,V}(parts)),
             C, C_val, C_parts...)
    else
        map!((val, parts...) -> Complex(
                 Dual{T,V,N}(real(val), ForwardDiff.Partials{N,V}(ntuple(i -> real(parts[i]), N))),
                 Dual{T,V,N}(imag(val), ForwardDiff.Partials{N,V}(ntuple(i -> imag(parts[i]), N)))),
             C, C_val, C_parts...)
    end
    C
end

# Overload of mul! and Base.:* methods resulting in a matrix of Dual/Complex{Dual}. Explicitly
# parametrize input types such that matrices of Dual/Complex{Dual} share the same tag. When mixed
# tags are present, revert to default (slow) Julia implemetation (happens rearely, and more complex
# workarounds would be necessary). Explicitly list Dual x Dual, Dual x Scalar and Scalar x Dual cases.

# 5-arguments mul!
function LinearAlgebra.mul!(C::AbstractMatrix{<:DualEltype{T,V,N}},
                            A::AbstractMatrix{<:DualEltype{T,V,N}},
                            B::AbstractMatrix{<:DualEltype{T,V,N}},
                            α::Number, β::Number) where {T,V,N}
    _dual_mul!(C, A, B, α, β)
end
function LinearAlgebra.mul!(C::AbstractMatrix{<:DualEltype{T,V,N}},
                            A::AbstractMatrix{<:DualEltype{T,V,N}},
                            B::AbstractMatrix{<:ScalarEltype},
                            α::Number, β::Number) where {T,V,N}
    _dual_mul!(C, A, B, α, β)
end
function LinearAlgebra.mul!(C::AbstractMatrix{<:DualEltype{T,V,N}},
                            A::AbstractMatrix{<:ScalarEltype},
                            B::AbstractMatrix{<:DualEltype{T,V,N}},
                            α::Number, β::Number) where {T,V,N}
    _dual_mul!(C, A, B, α, β)
end

# 3-arguments mul!
function LinearAlgebra.mul!(C::AbstractMatrix{<:DualEltype{T,V,N}},
                            A::AbstractMatrix{<:DualEltype{T,V,N}},
                            B::AbstractMatrix{<:DualEltype{T,V,N}}) where {T,V,N}
    _dual_mul!(C, A, B, true, false)
end
function LinearAlgebra.mul!(C::AbstractMatrix{<:DualEltype{T,V,N}},
                            A::AbstractMatrix{<:DualEltype{T,V,N}},
                            B::AbstractMatrix{<:ScalarEltype}) where {T,V,N}
    _dual_mul!(C, A, B, true, false)
end
function LinearAlgebra.mul!(C::AbstractMatrix{<:DualEltype{T,V,N}},
                            A::AbstractMatrix{<:ScalarEltype},
                            B::AbstractMatrix{<:DualEltype{T,V,N}}) where {T,V,N}
    _dual_mul!(C, A, B, true, false)
end

# Explicit Base.:* overloads
function Base.:*(A::AbstractMatrix{<:DualEltype{T,V,N}},
                 B::AbstractMatrix{<:DualEltype{T,V,N}}) where {T,V,N}
    S = promote_type(eltype(A), eltype(B))
    C = similar(A, S, size(A, 1), size(B, 2))
    mul!(C, A, B)
end
function Base.:*(A::AbstractMatrix{<:DualEltype{T,V,N}},
                 B::AbstractMatrix{<:ScalarEltype}) where {T,V,N}
    S = promote_type(eltype(A), eltype(B))
    C = similar(A, S, size(A, 1), size(B, 2))
    mul!(C, A, B)
end
function Base.:*(A::AbstractMatrix{<:ScalarEltype},
                 B::AbstractMatrix{<:DualEltype{T,V,N}}) where {T,V,N}
    S = promote_type(eltype(A), eltype(B))
    C = similar(B, S, size(A, 1), size(B, 2))
    mul!(C, A, B)
end
