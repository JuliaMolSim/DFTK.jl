"""
Perdew, Burke, Ernzerhof (DOI: 10.1103/PhysRevLett.77.3865)
"""
function energy_per_particle(::Val{:gga_c_pbe}, ρ, σ)
    β = 0.06672455060314922
    γ = (1 - log(2)) / π^2
    energy_per_particle_c_pbe(ρ, σ; β, γ)
end

"""
Xu, Goddard 2004 (DOI 10.1063/1.1771632)
"""
function energy_per_particle(::Val{:gga_c_xpbe}, ρ, σ)
    β = 0.089809  # Fitted constants, Table I
    α = 0.197363  # Fitted constants, Table I
    γ = β^2 / 2α
    energy_per_particle_c_pbe(ρ, σ; β, γ)
end

"""
Perdew, Ruzsinszky, Csonka and others 2008 (DOI 10.1103/physrevlett.100.136406)
"""
function energy_per_particle(::Val{:gga_c_pbe_sol}, ρ, σ)
    β = 0.046  # Page 3, left column below figure 1
    γ = (1 - log(2)) / π^2
    energy_per_particle_c_pbe(ρ, σ; β, γ)
end

"""
Constantin, Fabiano, Laricchia 2011 (DOI 10.1103/physrevlett.106.186406)
"""
function energy_per_particle(::Val{:gga_c_apbe}, ρ, σ)
    μ = 0.260   # p. 1, right column, bottom
    β = 3μ / π^2
    γ = (1 - log(2)) / π^2  # like in PBE
    energy_per_particle_c_pbe(ρ, σ; β, γ)
end

"""
del Campo, Gazqez, Trickey and others 2012 (DOI 10.1063/1.3691197)
"""
function energy_per_particle(::Val{:gga_c_pbe_mol}, ρ, σ)
    # β made to cancel self-interaction error in hydrogen
    β = 0.08384             # p. 4, right column, first paragraph
    γ = (1 - log(2)) / π^2  # like in PBE
    energy_per_particle_c_pbe(ρ, σ; β, γ)
end

"""
Sarmiento-Perez, Silvana, Marques 2015 (DOI 10.1021/acs.jctc.5b00529)
"""
function energy_per_particle(::Val{:gga_c_pbefe}, ρ, σ)
    β = 0.043                    # Fitted constants, Table I
    γ = 0.031090690869654895034  # Fitted constants, Table I
    energy_per_particle_c_pbe(ρ, σ; β, γ)
end

function energy_per_particle_c_pbe(ρ, σ; β, γ)
    # Spin-scaling factor with ζ spin polarization.
    # Yue Wang and John P. Perdew. Phys. Rev. B 43, 8911 (1991).
    # DOI 10.1103/PhysRevB.43.8911
    ϕ(ζ) = ((1+ζ)^(2/3) + (1-ζ)^(2/3))/2  # == 1 for non-spin-polarised

    # ε = UEG correlation energy per particle
    A(ε, ϕ³) = β/γ / expm1(-ε / (γ * ϕ³))  # (8)
    function H(ε, t², ϕ³)  # (7)
        At² = A(ε, ϕ³) * t²
        γ * ϕ³ * log(1 + β/γ * t² * (1 + At²) / (1 + At² + (At²)^2))
    end

    phi = #= ϕ(ζ) =# 1.0
    ε = energy_per_particle(Val(:lda_c_pw), ρ; improved_constants=true)
    t² = (1/12 * 3^(5/6) * π^(1/6))^2 * σ / (phi^2 * ρ^(7/3))  # page 2, left column, top
    ε + H(ε, t², phi)
end
