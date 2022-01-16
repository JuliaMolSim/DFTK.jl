"""
Perdew, Burke, Ernzerhof (DOI: 10.1103/PhysRevLett.77.3865)
"""
energy_per_particle(::Val{:gga_x_pbe}, ρ, σ) =
    energy_per_particle_x_pbe(ρ, σ, κ=0.8040, β=0.06672455060314922)

function energy_per_particle_x_pbe(ρ, σ; κ, β)
    μ = β * π^2 / 3
    pbe_x_f(s) = 1 + κ - κ^2 / (κ + μ * s^2)
    s = sqrt(σ/4) / ((ρ/2)^(4/3)) / 2cbrt(6π^2)
    energy_per_particle(Val(:lda_x), ρ) * pbe_x_f(s)
end
