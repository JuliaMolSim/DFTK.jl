"""
Perdew, Burke, Ernzerhof (DOI: 10.1103/PhysRevLett.77.3865)
"""
function energy_per_particle(::Val{:gga_x_pbe}, ρ, σ)
    β = 0.06672455060314922
    # μ eqn: below (12) in PBE paper
    energy_per_particle_x_pbe(ρ, σ; κ=0.8040, μ=β/3 * π^2)
end

"""
Zhang, Yang 1998 (DOI 10.1103/physrevlett.80.890)
"""
function energy_per_particle(::Val{:gga_x_pbe_r}, ρ, σ)
    β = 0.06672455060314922  # like in PBE
    energy_per_particle_x_pbe(ρ, σ; κ=1.245, μ=β/3 * π^2)
end

"""
Xu, Goddard 2004 (DOI 10.1063/1.1771632)
"""
function energy_per_particle(::Val{:gga_x_xpbe}, ρ, σ)
    energy_per_particle_x_pbe(ρ, σ; κ=0.91954, μ=0.23214)  # Table 1
end

"""
Perdew, Ruzsinszky, Csonka and others 2008 (DOI 10.1103/physrevlett.100.136406)
"""
function energy_per_particle(::Val{:gga_x_pbe_sol}, ρ, σ)
    energy_per_particle_x_pbe(ρ, σ; κ=0.8040, μ=10/81)  # μ given below equation (2)
end

"""
Constantin, Fabiano, Laricchia 2011 (DOI 10.1103/physrevlett.106.186406)
"""
function energy_per_particle(::Val{:gga_x_apbe}, ρ, σ)
    energy_per_particle_x_pbe(ρ, σ; κ=0.8040, μ=0.260)  # p. 1, right column, bottom
end

"""
del Campo, Gazqez, Trickey and others 2012 (DOI 10.1063/1.3691197)
"""
function energy_per_particle(::Val{:gga_x_pbe_mol}, ρ, σ)
    energy_per_particle_x_pbe(ρ, σ; κ=0.8040, μ=0.27583)  # p. 4, left column, bottom
end

"""
Sarmiento-Perez, Silvana, Marques 2015 (DOI 10.1021/acs.jctc.5b00529)
"""
function energy_per_particle(::Val{:gga_x_pbefe}, ρ, σ)
    energy_per_particle_x_pbe(ρ, σ; κ=0.437, μ=0.346)  # Table 1
end

function energy_per_particle_x_pbe(ρ, σ; κ, μ) # eqns refer to PBE paper
    pbe_x_f(s²) = 1 + κ - κ^2 / (κ + μ * s²)   # (14)
    # rₛ = cbrt(3 / (4π  * ρ))                 # page 2, left column, top
    # kF = cbrt(3π^2 * ρ)                      # page 2, left column, top
    # s  = sqrt(σ) / (2kF * ρ)                 # below (9)
    s² = σ / ( ρ^(4/3) * 2cbrt(3π^2) )^2

    energy_per_particle(Val(:lda_x), ρ) * pbe_x_f(s²)  # (10)
end
