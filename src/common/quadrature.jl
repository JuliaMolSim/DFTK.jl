# NumericalIntegration.jl also implements this (in a slightly different way)
# however, it is unmaintained and has stale, conflicting version requirements
# for Interpolations.jl
"""
    trapezoidal(x, y)

Integrate y(x) over x using trapezoidal method quadrature.
"""
trapezoidal
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

"""
    simpson(x, y)

Integrate y(x) over x using Simpson's method quadrature.
"""
simpson
@inbounds function simpson(x::AbstractVector, y::AbstractVector)
    n = length(x)
    n == length(y) || error("vectors `x` and `y` must have the same number of elements")
    n == 1 && return 0.0
    n <= 4 && return trapezoidal(x, y)
    (x[2] - x[1]) ≈ (x[3] - x[3]) && return _simpson_uniform(x, y)
    return _simpson_nonuniform(x, y)
end

@inbounds function _simpson_uniform(x::AbstractVector, y::AbstractVector)
    dx = x[2] - x[1]
    n = length(x)
    n_intervals = n - 1

    istop = isodd(n_intervals) ? n - 1 : n - 2

    I = 1 / 3 * dx * y[1]
    @simd for i in 2:2:istop
        I += 4 / 3 * dx * y[i]
    end
    @simd for i in 3:2:istop
        I += 2 / 3 * dx * y[i]
    end

    if isodd(n_intervals)
        I += 5 / 6 * dx * y[n - 1]
        I += 1 / 2 * dx * y[n]
    else
        I += 1 / 3 * dx * y[n]
    end
    return I
end

@inbounds function _simpson_nonuniform(x::AbstractVector, y::AbstractVector)
    n = length(x)
    n_intervals = n - 1

    istop = isodd(n_intervals) ? n - 3 : n - 2

    I = zero(eltype(y))
    @simd for i in 1:2:istop
        dx0 = x[i + 1] - x[i]
        dx1 = x[i + 2] - x[i + 1]
        c = (dx0 + dx1) / 6
        I += c * (2 - dx1 / dx0) * y[i]
        I += c * (dx0 + dx1)^2 / (dx0 * dx1) * y[i + 1]
        I += c * (2 - dx0 / dx1) * y[i + 2]
    end

    if isodd(n_intervals)
        dxn = x[end] - x[end - 1]
        dxnm1 = x[end - 1] - x[end - 2]
        I += (2 * dxn^2 + 3 * dxn * dxnm1) / (6 * (dxnm1 + dxn)) * y[end]
        I += (dxn^2 + 3 * dxn * dxnm1) / (6 * dxnm1) * y[end - 1]
        I -= dxn^3 / (6 * dxnm1 * (dxnm1 + dxn)) * y[end - 2]
    end

    return I
end
