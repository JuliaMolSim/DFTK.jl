using LoopVectorization

# NumericalIntegration.jl also implements this (in a slightly different way)
# however, it is unmaintained and has stale, conflicting version requirements
# for Interpolations.jl
"""
    trapezoidal(x, y)

Integrate y(x) over x using trapezoidal method quadrature.
"""
trapezoidal
@inbounds function trapezoidal(integrand, x::AbstractVector)
    n = length(x)
    n == 1 && return zero(promote_type(eltype(x), eltype(integrand(1, x[1]))))
    I = (x[2] - x[1]) * integrand(1, x[1])
    @turbo for i = 2:(n-1)
        # dx[i] + dx[i-1] = (x[i+1] - x[i]) + (x[i] - x[i-1])
        #                 = x[i+1] - x[i-1]
        I += (x[i+1] - x[i-1]) * integrand(i, x[i])
    end
    I += (x[n] - x[n-1]) * integrand(n, x[n])
    I / 2
end

"""
    simpson(x, y)

Integrate y(x) over x using Simpson's method quadrature.
"""
simpson
@inbounds function simpson(integrand, x::AbstractVector)
    n = length(x)
    n <= 4 && return trapezoidal(integrand, x)
    if (x[2] - x[1]) ≈ (x[3] - x[2])
        simpson_uniform(integrand, x)
    else
        simpson_nonuniform(integrand, x)
    end
end

@inbounds function simpson_uniform(integrand, x::AbstractVector)
    dx = x[2] - x[1]
    n = length(x)
    n_intervals = n - 1

    istop = isodd(n_intervals) ? n - 1 : n - 2

    I = 1 / 3 * dx * integrand(1, x[1])
    @turbo for i = 2:2:istop
        I += 4 / 3 * dx * integrand(i, x[i])
    end
    @turbo for i = 3:2:istop
        I += 2 / 3 * dx * integrand(i, x[i])
    end

    if isodd(n_intervals)
        I += 5 / 6 * dx * integrand(n, x[n-1])
        I += 1 / 2 * dx * integrand(n, x[n])
    else
        I += 1 / 3 * dx * integrand(n, x[n])
    end
    return I
end

@inbounds function simpson_nonuniform(integrand, x::AbstractVector)
    y = integrand
    END
    n = length(x)
    n_intervals = n - 1

    istop = isodd(n_intervals) ? n - 3 : n - 2

    I = zero(promote_type(eltype(x), eltype(y)))
    # This breaks when @turbo'd
    @simd for i = 1:2:istop
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
