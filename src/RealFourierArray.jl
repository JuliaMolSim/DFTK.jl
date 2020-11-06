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
mutable struct RealFourierArray{T <: Real,
                                TRealArray <: AbstractArray{T, 3},
                                TFourierArray <: AbstractArray{Complex{T}, 3}}
    basis::PlaneWaveBasis{T}
    _real::TRealArray
    _fourier::TFourierArray
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
function from_fourier(basis, fourier_part::AbstractArray{T}; check_real=true) where {T <: Complex}
    if check_real
        # Go through G vectors and check c_{-G} = (c_G)' (if both G and -G are in the grid)
        # arr[1] is G=0, arr[1] is G=1, arr[N] is G=-1.
        # So 1 -> 1, 2 -> N, ..., N -> 2
        reflect(i, N) = i == 1 ? 1 : N-i+2
        # Eg if N = 3, we get [0 1 -1], if N = 4, we get [0 1 2 -1] => we check to div(N+1,2)
        for i = 1:div(basis.fft_size[1]+1, 2)
            for j = 1:div(basis.fft_size[2]+1, 2)
                for k = 1:div(basis.fft_size[3]+1, 2)
                    err = abs(fourier_part[i, j, k] -
                              conj(fourier_part[reflect(i, end), reflect(j, end), reflect(k, end)]))
                    err > sqrt(eps(real(T))) && error("Input array not real")
                end
            end
        end
    end
    RealFourierArray(basis, similar(fourier_part, real(T)), fourier_part, RFA_only_fourier)
end

function Base.propertynames(array::RealFourierArray, private=false)
    ret = [:basis, :real, :fourier]
    private ? append!(ret, fieldnames(RealFourierArray)) : ret
end

function Base.getproperty(A::RealFourierArray{T, TReal, TFourier}, x::Symbol) where {T, TReal, TFourier}
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
        return A._real::TReal  # the compiler does not figure it out otherwise
    elseif x == :fourier
        if A._state == RFA_only_real
            r_to_G!(A._fourier, A.basis, complex.(A._real))
            setfield!(A, :_state, RFA_both)
        end
        return A._fourier::TFourier
    else
        getfield(A, x)
    end
end
function Base.setproperty!(::RealFourierArray, ::Any)
    error("RealFourierArray is intended to be read-only." *
          "(This can be bypassed by `setfield!` if you really want to).")
end

# Algebraic operations
function Base.:+(A::RealFourierArray, B::RealFourierArray)
    from_real(A.basis, A.real + B.real)
end
function Base.:-(A::RealFourierArray, B::RealFourierArray)
    from_real(A.basis, A.real - B.real)
end
function Base.:*(α::Number, A::RealFourierArray)
    from_real(A.basis, α * A.real)
end
Base.:*(A::RealFourierArray, α::Number) = α * A
Base.:/(A::RealFourierArray, α::Number) = A * inv(α)
