"""
Perdew, Burke, Ernzerhof (DOI: 10.1103/PhysRevLett.77.3865)
"""
function energy_per_particle(::Val{:gga_x_pbe}, ρ, σ)
    β = 0.06672455060314922
    μ = β/3 * π^2  # below (12) in PBE paper
    energy_per_particle_x_pbe(ρ, σ; κ=0.8040, μ)
end

"""
Perdew, Ruzsinszky, Csonka and others 2008 (DOI 10.1103/physrevlett.100.136406)
"""
function energy_per_particle(::Val{:gga_x_pbe_sol}, ρ, σ)
    energy_per_particle_x_pbe(ρ, σ; κ=0.8040, μ=10/81)  # μ given below equation (2)
end

function energy_per_particle_x_pbe(ρ, σ; κ, μ) # eqns refer to PBE paper
    pbe_x_f(s) = 1 + κ - κ^2 / (κ + μ * s^2)   # (14)
    # rₛ = cbrt(3 / (4π  * ρ))                 # page 2, left column, top
    # kF = cbrt(3π^2 * ρ)                      # page 2, left column, top
    # s  = sqrt(σ) / (2kF * ρ)                 # below (9)
    s = sqrt(σ) / ( ρ^(4/3) * 2cbrt(3π^2) )
    energy_per_particle(Val(:lda_x), ρ) * pbe_x_f(s)  # (10)
end
