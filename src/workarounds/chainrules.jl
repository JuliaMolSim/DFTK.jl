using ChainRulesCore
using Zygote: @adjoint  # TODO remove, once ChainRules rrules can overrule Zygote
import AbstractFFTs

function ChainRulesCore.rrule(::typeof(r_to_G), basis::PlaneWaveBasis, f_real::AbstractArray)
    f_fourier = r_to_G(basis, f_real)
    function r_to_G_pullback(Δf_fourier)
        ∂f_real = real(basis.opBFFT_unnormalized * Δf_fourier) * (sqrt(basis.model.unit_cell_volume) / length(basis.opFFT_unnormalized))
        return NoTangent(), @not_implemented("TODO ∂basis"), ∂f_real
    end
    return f_fourier, r_to_G_pullback
end

function ChainRulesCore.rrule(::typeof(G_to_r), basis::PlaneWaveBasis, f_fourier::AbstractArray; kwargs...)
    f_real = G_to_r(basis, f_fourier; kwargs...)
    function G_to_r_pullback(Δf_real)
        ∂f_fourier = (basis.opFFT * Δf_real) * (length(basis.opFFT) / (basis.model.unit_cell_volume))
        return NoTangent(), @not_implemented("TODO ∂basis"), ∂f_fourier
    end
    return f_real, G_to_r_pullback
end

function ChainRulesCore.rrule(::typeof(*), P::AbstractFFTs.ScaledPlan, x)
    y = P * x
    function mul_pullback(Δ)
      N = prod(size(x)[[P.p.region...]])
      ∂x = N * (P \ Δ)
      ∂P = Tangent{typeof(P)}(;scale=sum(Δ .* y))
      return NoTangent(), ∂P, ∂x 
    end
    return y, mul_pullback
end

# explicit Zygote.adjoint definitions are here bc Zygote.adjoint rules 
# take precedence over ChainRulesCore.rrule, even if rrule is more specific.
# Currently, Zygote defines an adjoint for *(P::AbstractFFTs.Plan, xs)
# that fails for ScaledPlan (which has no field `region`).
# TODO fix upstream and delete here
@adjoint function *(P::AbstractFFTs.ScaledPlan, xs)
    ys = P * xs
    return ys, function(Δ)
        N = prod(size(xs)[[P.p.region...]])
        return ((;scale=sum(Δ .* ys)), N * (P \ Δ))
    end
end

# workaround rrules for mpi: treat as noop
function ChainRulesCore.rrule(::typeof(mpi_sum), arr, comm)
    function mpi_sum_pullback(Δy)
        return NoTangent(), Δy, NoTangent()
    end
    return arr, mpi_sum_pullback
end

ChainRulesCore.@non_differentiable ElementPsp(::Any...)

# TODO delete
@adjoint (T::Type{<:SArray})(x...) = T(x...), y->(y,)
