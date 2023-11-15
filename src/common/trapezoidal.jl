# NumericalIntegration.jl also implements this (in a slightly different way)
# however, it is unmaintained and has stale, conflicting version requirements
# for Interpolations.jl
@inbounds function trapezoidal(x::AbstractVector, y::AbstractVector)
    n = length(x)
    n == length(y) || error("vectors `x` and `y` must have the same number of elements")
    n == 1 && return 0.0
    I = (x[2] - x[1]) * y[1]
    @simd for i in 2:(n-1)
        # dx[i] + dx[i - 1] = (x[i + 1] - x[i]) + (x[i] - x[i - 1])
        #                   = x[i + 1] - x[i - 1]
        I += (x[i + 1] - x[i - 1]) * y[i]
    end
    I += (x[n] - x[n - 1]) * y[n]
    I / 2
end
