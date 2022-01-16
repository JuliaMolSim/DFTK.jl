import ForwardDiff
import AbstractFFTs

# original PR by mcabbott: https://github.com/JuliaDiff/ForwardDiff.jl/pull/495

ForwardDiff.value(x::Complex{<:ForwardDiff.Dual}) = Complex(x.re.value, x.im.value)

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

function _apply_plan(p::AbstractFFTs.Plan, x::AbstractArray{<:Complex{<:ForwardDiff.Dual{T}}}) where T
    # TODO do we want x::AbstractArray{<:ForwardDiff.Dual{T}} too?
    xtil = p * ForwardDiff.value.(x)
    dxtils = ntuple(ForwardDiff.npartials(eltype(x))) do n
        p * ForwardDiff.partials.(x, n)
    end
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

# this is to avoid method ambiguities between these two:
#   _apply_plan(p::AbstractFFTs.Plan, x::AbstractArray{<:Complex{<:ForwardDiff.Dual{T}}}) where T
#   _apply_plan(p::AbstractFFTs.ScaledPlan{T,P,<:ForwardDiff.Dual}, x::AbstractArray) where {T,P}
function _apply_plan(p::AbstractFFTs.ScaledPlan{T,P,<:ForwardDiff.Dual}, x::AbstractArray{<:Complex{<:ForwardDiff.Dual{Tg}}}) where {T,P,Tg}
    _apply_plan(p.p, p.scale * x)
end

# DFTK setup specific

next_working_fft_size(::Type{<:ForwardDiff.Dual}, size::Int) = size

_fftw_flags(::Type{<:ForwardDiff.Dual}) = FFTW.MEASURE | FFTW.UNALIGNED

function build_fft_plans(T::Type{<:Union{ForwardDiff.Dual,Complex{<:ForwardDiff.Dual}}}, fft_size)
    tmp = Array{complex(T)}(undef, fft_size...) # TODO think about other Array types
    opFFT  = FFTW.plan_fft(tmp, flags=_fftw_flags(T))
    opBFFT = FFTW.plan_bfft(tmp, flags=_fftw_flags(T))

    ipFFT  = DummyInplace{typeof(opFFT)}(opFFT)
    ipBFFT = DummyInplace{typeof(opBFFT)}(opBFFT)
    # backward by inverting and stripping off normalizations
    ipFFT, opFFT, ipBFFT, opBFFT
end

# PlaneWaveBasis{<:Dual} contains dual-scaled fft, which means that the result f_fourier 
# must be able to hold complex dual numbers even if f_real is not dual
function r_to_G(basis::PlaneWaveBasis{T}, f_real::AbstractArray) where {T<:ForwardDiff.Dual}
    f_fourier = similar(f_real, complex(T))
    @assert length(size(f_real)) ∈ (3, 4)
    # this exploits trailing index convention
    for iσ = 1:size(f_real, 4)
        @views r_to_G!(f_fourier[:, :, :, iσ], basis, f_real[:, :, :, iσ])
    end
    f_fourier
end

# determine symmetry operations only from primal lattice values
function spglib_get_symmetry(lattice::Matrix{<:ForwardDiff.Dual}, atoms, magnetic_moments=[]; kwargs...)
    spglib_get_symmetry(ForwardDiff.value.(lattice), atoms, magnetic_moments; kwargs...)
end

function _is_well_conditioned(A::AbstractArray{<:ForwardDiff.Dual}; kwargs...)
    _is_well_conditioned(ForwardDiff.value.(A); kwargs...)
end


# other workarounds

# problem: ForwardDiff of norm of SVector gives NaN derivative at zero
# https://github.com/JuliaMolSim/DFTK.jl/issues/443#issuecomment-864930410
# solution: follow ChainRules custom frule for norm
# https://github.com/JuliaDiff/ChainRules.jl/blob/52a0eeadf8d19bff491f224517b7b064ce1ba378/src/rulesets/LinearAlgebra/norm.jl#L5
# TODO delete, once forward diff AD tools use ChainRules natively
function LinearAlgebra.norm(x::SVector{S,<:ForwardDiff.Dual{Tg,T,N}}) where {S,Tg,T,N}
    x_value = ForwardDiff.value.(x)
    y = norm(x_value)
    dy = ntuple(j->real(dot(x_value, ForwardDiff.partials.(x,j))) * pinv(y), N)
    ForwardDiff.Dual{Tg}(y, dy)
end

# problem: the derivative of 1/(1+exp(x)) = -exp(x) / (1+exp(x))^2.
# When x is too large, exp(x) = Inf and this is a NaN.
function Smearing.occupation(S::Smearing.FermiDirac, d::ForwardDiff.Dual{T}) where {T}
    x = ForwardDiff.value(d)
    if exp(x) > floatmax(typeof(x)) / 1e3
        ∂occ = -zero(x)
    else
        ∂occ = -exp(x) / (1 + exp(x))^2
    end
    ForwardDiff.Dual{T}(Smearing.occupation(S, x), ∂occ * ForwardDiff.partials(d))
end
