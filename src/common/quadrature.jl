# NumericalIntegration.jl also implements this (in a slightly different way)
# however, it is unmaintained and has stale, conflicting version requirements
# for Interpolations.jl
"""
    trapezoidal(x, y)

Integrate y(x) over x using trapezoidal method quadrature.
"""
trapezoidal
@inbounds function trapezoidal(integrand, x::AbstractVector, args...)
    n = length(x)
    Tint = eltype(integrand(1, x[1], args...))
    n == 1 && return zero(promote_type(eltype(x), Tint))
    I = (x[2] - x[1]) * integrand(1, x[1], args...)
    # Note: We used @turbo here before, but actually the allocation overhead
    #       needed to get all the data into an array is worse than what one gains
    #       with LoopVectorization
    @fastmath for i = 2:(n-1)
        # dx[i] + dx[i-1] = (x[i+1] - x[i]) + (x[i] - x[i-1])
        #                 = x[i+1] - x[i-1]
        I += (x[i+1] - x[i-1]) * integrand(i, x[i], args...)
    end
    I += (x[n] - x[n-1]) * integrand(n, x[n], args...)
    I / 2
end

"""
    simpson(x, y)

Integrate y(x) over x using Simpson's method quadrature.
"""
simpson
@inbounds function simpson(integrand, x::AbstractVector, args...)
    n = length(x)
    n <= 4 && return trapezoidal(integrand, x, args...)
    if (x[2] - x[1]) ≈ (x[3] - x[2])
        simpson_uniform(integrand, x, args...)
    else
        simpson_nonuniform(integrand, x, args...)
    end
end

@inbounds function simpson_uniform(integrand, x::AbstractVector, args...)
    dx = x[2] - x[1]
    n = length(x)
    n_intervals = n - 1

    istop = isodd(n_intervals) ? n - 1 : n - 2

    I = 1 / 3 * dx * integrand(1, x[1], args...)
    # Note: We used @turbo here before, but actually the allocation overhead
    #       needed to get all the data into an array is worse than what one gains
    #       with LoopVectorization
    @fastmath for i = 2:2:istop
        I += 4 / 3 * dx * integrand(i, x[i], args...)
    end
    @fastmath for i = 3:2:istop
        I += 2 / 3 * dx * integrand(i, x[i], args...)
    end

    if isodd(n_intervals)
        I += 5 / 6 * dx * integrand(n, x[n-1], args...)
        I += 1 / 2 * dx * integrand(n, x[n], args...)
    else
        I += 1 / 3 * dx * integrand(n, x[n], args...)
    end
    return I
end

@inbounds function simpson_nonuniform(integrand, x::AbstractVector, args...)
    n = length(x)
    n_intervals = n-1

    istop = isodd(n_intervals) ? n-3 : n-2

    Tint = eltype(integrand(1, x[1], args...))
    I = zero(promote_type(eltype(x), Tint))
    # Note: We used @simd here before, but actually the allocation overhead
    #       needed to get all the data into an array is worse than what one gains
    #       wtih vectorization
    @fastmath for i = 1:2:istop
        dx0 = x[i + 1] - x[i]
        dx1 = x[i+2] - x[i+1]
        c = (dx0 + dx1) / 6
        I += c * (2 - dx1 / dx0) * integrand(i, x[i], args...)
        I += c * (dx0 + dx1)^2 / (dx0 * dx1) * integrand(i+1, x[i+1], args...)

        I += c * (2 - dx0 / dx1) * integrand(i+2, x[i+2], args...)

    end

    if isodd(n_intervals)
        dxn = x[end] - x[end-1]
        dxnm1 = x[end - 1] - x[end-2]
        I += (2 * dxn^2 + 3 * dxn * dxnm1) / (6 * (dxnm1 + dxn)) * integrand(n, x[n], args...)
        I += (dxn^2 + 3 * dxn * dxnm1) / (6 * dxnm1) * integrand(n-1, x[n-1], args...)
        I -= dxn^3 / (6 * dxnm1 * (dxnm1 + dxn)) * integrand(n-2, x[n-2], args...)
    end

    return I
end
