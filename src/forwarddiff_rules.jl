import ForwardDiff
import AbstractFFTs

# original PR by mcabbott: https://github.com/JuliaDiff/ForwardDiff.jl/pull/495
# modified version: https://github.com/niklasschmitz/ForwardDiff.jl/blob/nfs/fft/src/fft.jl

ForwardDiff.value(x::Complex{<:ForwardDiff.Dual}) =
    Complex(x.re.value, x.im.value)

ForwardDiff.partials(x::Complex{<:ForwardDiff.Dual}, n::Int) =
    Complex(ForwardDiff.partials(x.re, n), ForwardDiff.partials(x.im, n))

ForwardDiff.npartials(x::Complex{<:ForwardDiff.Dual{T,V,N}}) where {T,V,N} = N
ForwardDiff.npartials(::Type{<:Complex{<:ForwardDiff.Dual{T,V,N}}}) where {T,V,N} = N

ForwardDiff.tagtype(x::Complex{<:ForwardDiff.Dual{T,V,N}}) where {T,V,N} = T
ForwardDiff.tagtype(::Type{<:Complex{<:ForwardDiff.Dual{T,V,N}}}) where {T,V,N} = T

# AbstractFFTs.complexfloat(x::AbstractArray{<:ForwardDiff.Dual}) = float.(x .+ 0im)
AbstractFFTs.complexfloat(x::AbstractArray{<:ForwardDiff.Dual}) = AbstractFFTs.complexfloat.(x)
AbstractFFTs.complexfloat(d::ForwardDiff.Dual{T,V,N}) where {T,V,N} = convert(ForwardDiff.Dual{T,float(V),N}, d) + 0im

AbstractFFTs.realfloat(x::AbstractArray{<:ForwardDiff.Dual}) = AbstractFFTs.realfloat.(x)
AbstractFFTs.realfloat(d::ForwardDiff.Dual{T,V,N}) where {T,V,N} = convert(ForwardDiff.Dual{T,float(V),N}, d)

for plan in [:plan_fft, :plan_ifft, :plan_bfft]
    @eval begin

        AbstractFFTs.$plan(x::AbstractArray{<:ForwardDiff.Dual}, region=1:ndims(x); kwargs...) =
            AbstractFFTs.$plan(ForwardDiff.value.(x) .+ 0im, region; kwargs...)

        AbstractFFTs.$plan(x::AbstractArray{<:Complex{<:ForwardDiff.Dual}}, region=1:ndims(x); kwargs...) =
            AbstractFFTs.$plan(ForwardDiff.value.(x), region; kwargs...)

    end
end

# rfft only accepts real arrays
AbstractFFTs.plan_rfft(x::AbstractArray{<:ForwardDiff.Dual}, region=1:ndims(x); kwargs...) =
    AbstractFFTs.plan_rfft(ForwardDiff.value.(x), region; kwargs...)

for plan in [:plan_irfft, :plan_brfft]  # these take an extra argument, only when complex?
    @eval begin

        AbstractFFTs.$plan(x::AbstractArray{<:ForwardDiff.Dual}, region=1:ndims(x); kwargs...) =
            AbstractFFTs.$plan(ForwardDiff.value.(x) .+ 0im, region; kwargs...)

        AbstractFFTs.$plan(x::AbstractArray{<:Complex{<:ForwardDiff.Dual}}, d::Integer, region=1:ndims(x); kwargs...) =
            AbstractFFTs.$plan(ForwardDiff.value.(x), d, region; kwargs...)

    end
end

for P in [:Plan, :ScaledPlan]  # need ScaledPlan to avoid ambiguities
    @eval begin

        Base.:*(p::AbstractFFTs.$P, x::AbstractArray{<:ForwardDiff.Dual}) =
            _apply_plan(p, x)

        Base.:*(p::AbstractFFTs.$P, x::AbstractArray{<:Complex{<:ForwardDiff.Dual}}) =
            _apply_plan(p, x)

        LinearAlgebra.mul!(Y::AbstractArray, p::AbstractFFTs.$P, X::AbstractArray{<:ForwardDiff.Dual}) = 
            (Y .= _apply_plan(p, X))
        
        LinearAlgebra.mul!(Y::AbstractArray, p::AbstractFFTs.$P, X::AbstractArray{<:Complex{<:ForwardDiff.Dual}}) =
            (Y .= _apply_plan(p, X))
    end
end

LinearAlgebra.mul!(Y::AbstractArray{<:Complex{<:ForwardDiff.Dual}}, p::AbstractFFTs.ScaledPlan{T,P,<:ForwardDiff.Dual}, X::AbstractArray{<:ComplexF64}) where {T,P} =
    (Y .= _apply_plan(p, X))

function _apply_plan(p::AbstractFFTs.Plan, x::AbstractArray)
    xtil = p * ForwardDiff.value.(x)
    dxtils = ntuple(ForwardDiff.npartials(eltype(x))) do n
        p * ForwardDiff.partials.(x, n)
    end
    T = ForwardDiff.tagtype(eltype(x))
    map(xtil, dxtils...) do val, parts...
        Complex(
            ForwardDiff.Dual{T}(real(val), map(real, parts)),
            ForwardDiff.Dual{T}(imag(val), map(imag, parts)),
        )
    end
end

function _apply_plan(p::AbstractFFTs.ScaledPlan{T,P,<:ForwardDiff.Dual}, x::AbstractArray) where {T,P}
    _apply_plan(p.p, p.scale * x) # for when p.scale is Dual, need out-of-place
end

###
### DFTK setup specific
###

next_working_fft_size(::Type{<:ForwardDiff.Dual}, size) = size

_fftw_flags(::Type{<:ForwardDiff.Dual}) = FFTW.MEASURE | FFTW.UNALIGNED

# *** COPIED from fft_generic.jl *** TODO refactor
# A dummy wrapper around an out-of-place FFT plan to make it appear in-place
# This is needed for some generic FFT implementations, which do not have in-place plans
struct DummyInplace{opFFT}
    fft::opFFT
end
LinearAlgebra.mul!(Y, p::DummyInplace, X) = (Y .= mul!(similar(X), p.fft, X))
LinearAlgebra.ldiv!(Y, p::DummyInplace, X) = (Y .= ldiv!(similar(X), p.fft, X))

import Base: *, \, length
*(p::DummyInplace, X) = p.fft * X
\(p::DummyInplace, X) = p.fft \ X
length(p::DummyInplace) = length(p.fft)

function build_fft_plans(T::Type{<:Union{ForwardDiff.Dual,Complex{<:ForwardDiff.Dual}}}, fft_size)
    tmp = Array{Complex{T}}(undef, fft_size...)
    opFFT  = FFTW.plan_fft(tmp, flags=_fftw_flags(T))
    opBFFT = FFTW.plan_bfft(tmp, flags=_fftw_flags(T))

    ipFFT  = DummyInplace{typeof(opFFT)}(opFFT)
    ipBFFT = DummyInplace{typeof(opBFFT)}(opBFFT)
    # backward by inverting and stripping off normalizations
    ipFFT, opFFT, ipBFFT, opBFFT
end

function r_to_G(basis::PlaneWaveBasis{T}, f_real::AbstractArray) where {T<:ForwardDiff.Dual}
    f_fourier = similar(f_real, complex(T))
    @assert length(size(f_real)) ∈ (3, 4)
    # this exploits trailing index convention
    for iσ = 1:size(f_real, 4)
        @views r_to_G!(f_fourier[:, :, :, iσ], basis, f_real[:, :, :, iσ])
    end
    f_fourier
end

###
### other workarounds
###

# problem: ForwardDiff of norm of SVector gives NaN derivative at zero
# https://github.com/JuliaMolSim/DFTK.jl/issues/443#issuecomment-864930410
# solution: follow ChainRules custom frule for norm
# https://github.com/JuliaDiff/ChainRules.jl/blob/52a0eeadf8d19bff491f224517b7b064ce1ba378/src/rulesets/LinearAlgebra/norm.jl#L5
# TODO delete, once forward diff AD tools use ChainRules natively
function LinearAlgebra.norm(x::SVector{S,<:ForwardDiff.Dual}) where {S}
    T = ForwardDiff.tagtype(eltype(x))
    dx = ForwardDiff.partials.(x)
    y = norm(ForwardDiff.value.(x))
    dy = real(dot(ForwardDiff.value.(x), dx)) * pinv(y)
    ForwardDiff.Dual{T}(y, dy)
end
