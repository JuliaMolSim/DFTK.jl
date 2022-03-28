"""Functions to evaluate exp(), cos(), or sin() involving 2π"""
function cis2pi(z)
    return cispi(2*z)
end

function sin2pi(z)
    return sinpi(2*z)
end

function cos2pi(z)
    return cospi(2*z)
end