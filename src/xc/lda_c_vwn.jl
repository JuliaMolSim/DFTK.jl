"""
LDA correlation according to Vosko Wilk,and Nusair, (DOI 10.1139/p80-159)
"""
function lda_c_vwn!(ρ; E=nothing, Vρ=nothing)
    x0  = sqrt(5913592351)
    x1  = 2^(2 / 3) * 3^(1 / 6)
    x2  = pi^(1 / 6)
    x3  = ρ .^ (1 / 6)
    x4  = x2 .* x3
    x5  = 468052052839577158149 * x0 / 928058053502134909615000000
    x6  = 6^(1 / 3)
    x7  = 12500 * x4 .* x6
    x8  = pi^(1 / 3)
    x9  = ρ .^ (1 / 3)
    x10 = x8 .* x9
    x11 = 1 ./ (46593 * x1 .* x10 + x7 + 323380 * sqrt(pi) * sqrt.(ρ))
    x12 = x1 ./ x2
    x13 = x12 ./ x3
    x14 = x13 .+ 46593 / 12500
    x15 = x6 ./ x8
    x16 = x15 ./ (2 * x9)
    x17 = 46593 * x13 / 25000 .+ x16 .+ 16169 / 1250
    x18 = 1 ./ x17
    x19 = x13 / 2 .+ 5249 / 50000
    x20 = x19 .^ 2
    x21 = x14 .^ (-2)
    x22 = x12 ./ ρ .^ (7 / 6)
    x23 = x15 ./ (6 * ρ .^ (4 / 3))
    x24 = (15531 * x22 / 50000 + x23) ./ x17 .^ 2

    if E !== nothing
        @. E .=
            x5 * atan(x0 * x4 / (12500 * x1 + 46593 * x4)) +
            310907 * log(x11 * x7) / 10000000 +
            76037485627899 * log(x11 * x4 * (25000 * x1 + 5249 * x4)^2 / 100000) /
            78468213432500000
    end
    if Vρ !== nothing
        @. Vρ .=
            x5 * atan(x0 / (12500 * x14)) +
            ρ * (
                310907 * 6^(2 / 3) * x10 * x17 * (x16 * x24 - x18 * x23) / 30000000 +
                76037485627899 * x17 * (-x18 * x19 * x22 / 6 + x20 * x24) /
                (78468213432500000 * x20) +
                156017350946525719383 * x21 * x22 /
                (3923410671625000000000 * (5913592351 * x21 / 156250000 + 1))
            ) +
            310907 * log(x16 * x18) / 10000000 +
            76037485627899 * log(x18 * x20) / 78468213432500000
    end

    (E=E, Vρ=Vρ)
end
