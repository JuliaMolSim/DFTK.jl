"""
Perdew, Wang correlation from 1992 (10.1103/PhysRevB.45.13244)
"""
function energy_per_particle(::Val{:lda_c_pw}, ρ; improved_constants=false)
    α₁ = (0.21370,  0.20548,  0.11125)
    β₁ = (7.5957,  14.1189,  10.357)
    β₂ = (3.5876,   6.1977,   3.6231)
    β₃ = (1.6382,   3.3662,   0.88026)
    β₄ = (0.49294,  0.62517,  0.49671)
    # constant p = 1 is hard-coded in the expression for G below

    if !improved_constants  # Constants as given in original publication
        A   = (0.031091, 0.015545, 0.016887)
        f′′ = 1.709921  # f′′(0)
    else  # Modified constants, computed at improved accuracy
        A   = (0.0310907, 0.01554535, 0.0168869)
        f′′ = 8 / (9 * 2(cbrt(2) - 1))  # f′′(0)
    end

    energy_per_particle_c_pw(ρ; A, α₁, β₁, β₂, β₃, β₄, f′′)
end

function energy_per_particle_c_pw(ρ; A, α₁, β₁, β₂, β₃, β₄, f′′)
    function G(sqrt_rₛ, A, α₁, β₁, β₂, β₃, β₄)  # (10) with p = 1 hard-coded
        denom = β₁ * sqrt_rₛ + β₂ * sqrt_rₛ^2 + β₃ * sqrt_rₛ^3 + β₄ * sqrt_rₛ^4
        -2A * (1 + α₁*sqrt_rₛ^2) * log(1 + 1 / (2A * denom) )
    end

    # equation (9)
    f(ζ) = ((1+ζ)^(4/3) + (1-ζ)^(4/3) - 2)/(2^(4/3) - 2)  # == 0 for non-spin-polarised

    ε_0(rₛ) =  G(sqrt(rₛ), A[1], α₁[1], β₁[1], β₂[1], β₃[1], β₄[1])  # ε_c(rₛ, 0)
    ε_1(rₛ) =  G(sqrt(rₛ), A[2], α₁[2], β₁[2], β₂[2], β₃[2], β₄[2])  # ε_c(rₛ, 1)
    α(rₛ)   = -G(sqrt(rₛ), A[3], α₁[3], β₁[3], β₂[3], β₃[3], β₄[3])  # α_c(rₛ)

    # equation (8)
    ε(rₛ, ζ) = ε_0(rₛ) + α(rₛ) * f(ζ)/f′′ * (1 - ζ^4) + (ε_1(rₛ) - ε_0(rₛ)) * f(ζ) * ζ^4

    rₛ = cbrt(3 / (4π  * ρ))  # equation (1)
    ε(rₛ, #= ζ = =# 0)
end
