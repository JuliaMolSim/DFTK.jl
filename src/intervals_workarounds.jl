import IntervalArithmetic: Interval, mid
import SpecialFunctions: erfc

# Monkey-patch a few functions for Intervals
# ... this is far from proper and a bit specific for our use case here
# (that's why it's not contributed upstream).
# should be done e.g. by changing  the rounding mode ...
erfc(i::Interval) = Interval(prevfloat(erfc(i.lo)), nextfloat(erfc(i.hi)))

function compute_fft_size(lattice::AbstractMatrix{T}, Ecut; kwargs...) where T <: Interval
    # This is done to avoid a call like ceil(Int, ::Interval)
    # in the above implementation of compute_fft_size,
    # where it is in general cases not clear, what to do.
    # In this case we just want a reasonable number for Gmax,
    # so replacing the intervals in the lattice with
    # their midpoints should be good.
    compute_fft_size(mid.(lattice), Ecut; kwargs...)
end

function local_potential_fourier(el::ElementCohenBergstresser, q::T) where {T <: Interval}
    lor = round(q.lo, digits=5)
    hir = round(q.hi, digits=5)
    @assert iszero(round(lor - hir, digits=3))
    T(local_potential_fourier(el, mid(q)))
end
