"""
LDA Slater exchange (DOI: 10.1017/S0305004100016108 and 10.1007/BF01340281)
"""
function energy_per_particle(::Val{:lda_x}, ρ::T) where {T}
    # Severe numerical issues if this is not done at least at Float64
    W = promote_type(Float64, T)
    T(-3/W(4) * cbrt(3/W(π) * ρ))
end
