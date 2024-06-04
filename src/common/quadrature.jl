# NumericalIntegration.jl also implements this (in a slightly different way)
# however, it is unmaintained and has stale, conflicting version requirements
# for Interpolations.jl
"""
Integrate the `integrand` function using the nodal points `x`
using the trapezoidal rule.
The function will be called as `integrand(i, x[i])` for each integrand
point `i` (not necessarily in order).
"""
trapezoidal
@inbounds function trapezoidal(integrand, x::AbstractVector)
    n = length(x)
    Tint = eltype(integrand(1, x[1]))
    n == 1 && return zero(promote_type(eltype(x), Tint))
    I = (x[2] - x[1]) * integrand(1, x[1])
    # Note: We used @turbo here before, but actually the allocation overhead
    #       needed to get all the data into an array is worse than what one gains
    #       with LoopVectorization
    @fastmath @simd for i = 2:(n-1)
        # dx[i] + dx[i-1] = (x[i+1] - x[i]) + (x[i] - x[i-1])
        #                 = x[i+1] - x[i-1]
        I += @inline (x[i+1] - x[i-1]) * integrand(i, x[i])
    end
    I += (x[n] - x[n-1]) * integrand(n, x[n])
    I / 2
end

"""
Integrate a function represented by the nodal points and function values
given by the arrays `x`, `y`. Note the order (`y` comes first).
"""
trapezoidal(y::AbstractArray, x::AbstractArray) = trapezoidal((i, xi) -> y[i], x)

"""
Integrate the `integrand` function using the nodal points `x` using Simpson's rule.
The function will be called as `integrand(i, x[i])` for each integrand
point `i` (not necessarily in order).
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

"""
Integrate a function represented by the nodal points and function values
given by the arrays `x`, `y`. Note the order (`y` comes first).
"""
simpson(y::AbstractArray, x::AbstractArray) = simpson((i, xi) -> y[i], x)

@inbounds function simpson_uniform(integrand, x::AbstractVector)
    dx = x[2] - x[1]
    n = length(x)
    n_intervals = n - 1

    istop = isodd(n_intervals) ? n - 1 : n - 2

    I = 1 / 3 * dx * integrand(1, x[1])
    # Note: We used @turbo here before, but actually the allocation overhead
    #       needed to get all the data into an array is worse than what one gains
    #       with LoopVectorization
    @fastmath @simd for i = 2:2:istop
        I += @inline 4 / 3 * dx * integrand(i, x[i])
    end
    @fastmath @simd for i = 3:2:istop
        I += @inline 2 / 3 * dx * integrand(i, x[i])
    end

    if isodd(n_intervals)
        I += 5 / 6 * dx * integrand(n-1, x[n-1])
        I += 1 / 2 * dx * integrand(n, x[n])
    else
        I += 1 / 3 * dx * integrand(n, x[n])
    end
    return I
end

@inbounds function simpson_nonuniform(integrand, x::AbstractVector)
    n = length(x)
    n_intervals = n-1

    istop = isodd(n_intervals) ? n-3 : n-2

    Tint = eltype(integrand(1, x[1]))
    I = zero(promote_type(eltype(x), Tint))
    @fastmath @simd for i = 1:2:istop
        dx0 = x[i + 1] - x[i]
        dx1 = x[i+2] - x[i+1]
        c = (dx0 + dx1) / 6
        @inline begin
            I += c * (2 - dx1 / dx0) * integrand(i, x[i])
            I += c * (dx0 + dx1)^2 / (dx0 * dx1) * integrand(i+1, x[i+1])
            I += c * (2 - dx0 / dx1) * integrand(i+2, x[i+2])
        end

    end

    if isodd(n_intervals)
        dxn = x[end] - x[end-1]
        dxnm1 = x[end - 1] - x[end-2]
        I += (2 * dxn^2 + 3 * dxn * dxnm1) / (6 * (dxnm1 + dxn)) * integrand(n, x[n])
        I += (dxn^2 + 3 * dxn * dxnm1) / (6 * dxnm1) * integrand(n-1, x[n-1])
        I -= dxn^3 / (6 * dxnm1 * (dxnm1 + dxn)) * integrand(n-2, x[n-2])
    end

    return I
end
