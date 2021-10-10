import SpecialFunctions: erfc
import IntervalArithmetic: Interval, mid

# Monkey-patch a few functions for Intervals
# ... this is far from proper and a bit specific for our use case here
# (that's why it's not contributed upstream).
# should be done e.g. by changing  the rounding mode ...
erfc(i::Interval) = Interval(prevfloat(erfc(i.lo)), nextfloat(erfc(i.hi)))

function compute_Glims_fast(lattice::AbstractMatrix{<:Interval}, args...; kwargs...)
    # This is done to avoid a call like ceil(Int, ::Interval)
    # in the above implementation of compute_fft_size,
    # where it is in general cases not clear, what to do.
    # In this case we just want a reasonable number for Gmax,
    # so replacing the intervals in the lattice with
    # their midpoints should be good.
    compute_Glims_fast(mid.(lattice), args...; kwargs...)
end
function compute_Glims_precise(::AbstractMatrix{<:Interval}, args...; kwargs...)
    error("fft_size_algorithm :precise not supported with intervals")
end

function _is_well_conditioned(A::AbstractArray{<:Interval}; kwargs...)
    # This check is used during the lattice setup, where it frequently fails with intervals
    # (because doing an SVD with intervals leads to a large overestimation of the rounding error)
    _is_well_conditioned(mid.(A); kwargs...)
end

function symmetry_operations(lattice::AbstractMatrix{<:Interval}, atoms, magnetic_moments=[];
                             tol_symmetry=max(1e-5, maximum(radius, lattice)))
    @assert tol_symmetry < 1e-2
    symmetry_operations(mid.(lattice), atoms, magnetic_moments; tol_symmetry)
end

function local_potential_fourier(el::ElementCohenBergstresser, q::T) where {T <: Interval}
    lor = round(q.lo, digits=5)
    hir = round(q.hi, digits=5)
    @assert iszero(round(lor - hir, digits=3))
    T(local_potential_fourier(el, mid(q)))
end
