"""
Perdew, Burke, Ernzerhof (DOI: 10.1103/PhysRevLett.77.3865)
"""
function energy_per_particle(::Val{:gga_c_pbe}, ρ, σ)
    β = 0.06672455060314922
    γ = (1 - log(2.0)) / π^2

    # ε = UEG correlation energy per particle
    A(ε, ϕ³) = β/γ / expm1(-εc / (γ * ϕ³))
    ϕ(ζ) = ((1+ζ)^(2/3) + (1-ζ)^(2/3))/2  # == 1 for non-spin-polarised
    function H(ε, t², ϕ³)
        At² = A(ε, ϕ³) * t²
        γ * ϕ³ * log(1 + β/γ * t² * (1 + At²) / (1 + At² + (At²)^2))
    end

    phi = #= ϕ(ζ) =# 1.0
    ε = energy_per_particle(::Val{:lda_c_pw}, ρ)
    t² = (1/12 * 3^(5/6) * π^(1/6))^2 * σ / (phi^2 * ρ^(7/3))
    ε + H(ε, t², phi)
end
