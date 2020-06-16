# Lazy structure for an array whose real and fourier part can be accessed by A.real and A.fourier
# Used to represent potentials and densities
# For normalization conventions, see PlaneWaveBasis.jl

import Base.getproperty, Base.setproperty!, Base.propertynames
import Base.*, Base.+, Base./
import Base.eltype, Base.size, Base.length

## TODO these functions are minimally optimized for the moment
## TODO this always allocates both real and fourier parts, which might be a bit wasteful
## TODO implement broadcasting. This seems to be non-trivial, so I just implement a few simple methods
## TODO implement compressed representation in Fourier space of real arrays

@enum RFA_state RFA_only_real RFA_only_fourier RFA_both

"""
A structure to facilitate manipulations of an array of real-space type T, in both real
and fourier space. Create with `from_real` or `from_fourier`, and access
with `A.real` and `A.fourier`.
"""
mutable struct RealFourierArray{T <: Real}
    basis::PlaneWaveBasis{T}
    _real::AbstractArray{T, 3}
    _fourier::AbstractArray{Complex{T}, 3}
    _state::RFA_state
end
# Type of the real part
Base.eltype(::RealFourierArray{T}) where {T} = T
Base.size(array::RealFourierArray) = size(array._real)
Base.size(array::RealFourierArray, i) = size(array._real, i)
Base.length(array::RealFourierArray) = length(array._real)

# Constructors
RealFourierArray(basis, real, fourier) = RealFourierArray(basis, real, fourier, RFA_both)
function RealFourierArray(basis::PlaneWaveBasis{T}) where {T}
    from_real(basis, zeros(T, basis.fft_size...))
end

function from_real(basis, real_part::AbstractArray{T}) where {T <: Real}
    RealFourierArray(basis, real_part, similar(real_part, complex(T)), RFA_only_real)
end
function from_fourier(basis, fourier_part::AbstractArray{T}; check_real=false) where {T <: Complex}
    if check_real
        error("Not implemented yet")
        # Go through G vectors and check c_{-G} = (c_G)' (if both G and -G are in the grid)
        # Should be reasonably fast so we can make it the default
    end
    RealFourierArray(basis, similar(fourier_part, real(T)), fourier_part, RFA_only_fourier)
end

function Base.propertynames(array::RealFourierArray, private=false)
    ret = [:basis, :real, :fourier]
    private ? append!(ret, fieldnames(RealFourierArray)) : ret
end

function Base.getproperty(A::RealFourierArray, x::Symbol)
    if x == :real
        if A._state == RFA_only_fourier
            r = G_to_r(A.basis, A._fourier)
            if eltype(A) <: Real
                A._real .= real(r)
            else
                A._real .= r
            end
            setfield!(A, :_state, RFA_both)
        end
        return A._real
    elseif x == :fourier
        if A._state == RFA_only_real
            r_to_G!(A._fourier, A.basis, complex.(A._real))
            setfield!(A, :_state, RFA_both)
        end
        return A._fourier
    else
        getfield(A, x)
    end
end
function Base.setproperty!(A::RealFourierArray, x)
    error("RealFourierArray is intended to be read-only." *
          "(This can be bypassed by `setfield!` if you really want to).")
end

# Algebraic operations
function Base.:+(A::RealFourierArray, B::RealFourierArray) where T
    @assert A.basis == B.basis
    from_real(A.basis, A.real + B.real)
end
function Base.:-(A::RealFourierArray, B::RealFourierArray) where T
    @assert A.basis == B.basis
    from_real(A.basis, A.real - B.real)
end
function Base.:*(α::Number, A::RealFourierArray)
    from_real(A.basis, α * A.real)
end
Base.:*(A::RealFourierArray, α::Number) = α * A
Base.:/(A::RealFourierArray, α::Number) = A * inv(α)
