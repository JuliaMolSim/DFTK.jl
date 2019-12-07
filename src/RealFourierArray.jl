# Lazy structure for an array whose real and fourier part can be accessed by A.real and A.fourier
# For normalization conventions, see PlaneWaveBasis.jl

import Base.getproperty, Base.setproperty!, Base.fieldnames
import Base.*, Base.+, Base./
import Base.eltype

## TODO these functions are minimally optimized for the moment
## TODO this always allocates both real and fourier parts, which might be a bit wasteful
## TODO implement broadcasting. This seems to be non-trivial, so I just implement a few simple methods
## TODO implement compressed representation in Fourier space of real arrays

@enum RFA_state RFA_only_real RFA_only_fourier RFA_both

# TODO make this concrete by parametrizing on the type of the arrays
"""
A structure to facilitate manipulations of an array of type T in both real
and fourier space. Create with `from_real` or `from_fourier`, and access
with `A.real` and `A.fourier`.
"""
mutable struct RealFourierArray{Treal <: Real, T <: Union{Treal, Complex{Treal}}}
# Treal is the underlying real type
# T is the type of the array in real space
    basis::PlaneWaveBasis{Treal}
    _real::AbstractArray{T, 3}
    _fourier::AbstractArray{Complex{Treal}, 3}
    _state::RFA_state
end
# Type of the real part
Base.eltype(A::RealFourierArray{Treal, T}) where {Treal, T} = T

# Constructors
RealFourierArray(basis, real, fourier) = RealFourierArray(basis, real, fourier, 2)
function RealFourierArray(basis::PlaneWaveBasis{T}; iscomplex=false) where {T}
    from_real(basis, zeros(iscomplex ? complex(T) : T, basis.fft_size...))
end

function from_real(basis, real_part::AbstractArray{T}) where {T <: Number}
    RealFourierArray{real(T), T}(basis, real_part, similar(real_part, complex(T)), RFA_only_real)
end
function from_fourier(basis, fourier_part::AbstractArray{T}; isreal=false) where {T <: Complex}
    if isreal
        RealFourierArray{real(T), real(T)}(basis, similar(fourier_part, real(T)), fourier_part, RFA_only_fourier)
    else
        RealFourierArray{real(T), T}(basis, similar(fourier_part), fourier_part, RFA_only_fourier)
    end
end

check_real(A::RealFourierArray{Treal, T}) where {Treal, T <: Real} = nothing
check_real(A::RealFourierArray) = check_real(A.real)

# lie blatantly to the user, as recommended by the docs
Base.fieldnames(RealFourierArray) = (:basis, :real, :fourier)

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
