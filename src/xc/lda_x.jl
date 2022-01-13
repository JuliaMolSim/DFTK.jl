using ForwardDiff
"""
LDA Slater exchange (DOI: 10.1017/S0305004100016108 and 10.1007/BF01340281)
"""
E_lda_x(ρ) = 3/4 * cbrt(3/π * ρ)
