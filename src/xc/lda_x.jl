"""
LDA Slater exchange (DOI: 10.1017/S0305004100016108 and 10.1007/BF01340281)
"""
function energy_per_particle(::Val{:lda_x}, ρ) 
    #@warn "lda_x energy per particle"
    if ρ <= 1e-8
        return 0.0
    end
    -3/4 * cbrt(3/π * ρ)
end
