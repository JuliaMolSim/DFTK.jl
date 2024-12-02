"""Function to compute exp(2π i x)"""
cis2pi(x) = cispi(2x)
function cis2pi(x::AbstractFloat)
    # Special case when 2x is an integer, as exp(n*π*i) = +- 1. Saves expensive
    # exponential evaluations when called repeatedly
    if isinteger(2x)
        isodd(2x) ? result = -Complex{typeof(x)}(1, 0) : result = Complex{typeof(x)}(1, 0)
    else
        result = cispi(2x)
    end
    result
end

"""Function to compute sin(2π x)"""
sin2pi(x) = sinpi(2x)

"""Function to compute cos(2π x)"""
cos2pi(x) = cospi(2x)