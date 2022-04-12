"""
VWN5 LDA correlation according to Vosko, Wilk, and Nusair, (DOI 10.1139/p80-159).
"""
function energy_per_particle(::Val{:lda_c_vwn}, ρ::T) where {T}
    # From https://math.nist.gov/DFTdata/atomdata/node5.html
    A   = T( 0.0310907)
    x0  = T(-0.10498)
    b   = T( 3.72744)
    c   = T( 12.9352)
    rₛ  = cbrt(3/(T(4π)*ρ)) # τ in the above link
    x   = sqrt(rₛ)
    Xx  = x^2 + b*x + c
    Xx0 = x0^2 + b*x0 + c
    Q   = sqrt(4c-b^2)
    A * (log(x^2 / Xx) + 2b/Q*atan(Q/(2x+b)) - b*x0/Xx0*(log((x-x0)^2/Xx) + 2*(b+2x0)/Q*atan(Q/(2x+b))))
end
