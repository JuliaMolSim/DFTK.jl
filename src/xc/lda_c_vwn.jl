using ForwardDiff
"""
LDA correlation according to Vosko, Wilk, and Nusair, (DOI 10.1139/p80-159)
"""
function E_lda_c_vwn(ρ)
    # Ugly automatically generated code. TODO code from the equations
    x0  = sqrt(5913592351)
    x1  = 2^(2/3)*3^(1/6)
    x2  = pi^(1/6)
    x3  = ρ^(1/6)
    x4  = x2*x3
    x5  = 468052052839577158149*x0/928058053502134909615000000
    x6  = 6^(1/3)
    x7  = 12500*x4*x6
    x8  = pi^(1/3)
    x9  = ρ^(1/3)
    x10 = x8*x9
    x11 = 1 / (46593*x1.*x10 + x7 + 323380*sqrt(pi)*sqrt(ρ))

    x5*atan(x0*x4/(12500*x1 + 46593*x4)) + 310907*log(x11*x7)/10000000 + 76037485627899*log(x11*x4*(25000*x1 + 5249*x4)^2/100000)/78468213432500000
end
