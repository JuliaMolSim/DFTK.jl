"""Function to compute exp(2π i x)"""
cis2pi(x) = cispi(2x)
function cis2pi(x::T) where {T <: AbstractFloat}
    # Special case when 2x is an integer, as exp(n*π*i) = +- 1. Saves expensive
    # exponential evaluations when called repeatedly
    if isinteger(2x)
        return isodd(2x) ? -one(complex(T)) : one(complex(T))
    else
        return cispi(2x)
    end
end

"""Function to compute sin(2π x)"""
sin2pi(x) = sinpi(2x)

"""Function to compute cos(2π x)"""
cos2pi(x) = cospi(2x)