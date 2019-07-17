"""
LDA Slater exchange (DOI: 10.1017/S0305004100016108 and 10.1007/BF01340281)
"""
function lda_x!(ρ; α=1, E=nothing, Vρ=nothing)
    tmp = @. -α * cbrt(3/π * ρ)

    if E !== nothing
        E .= 3/4 .* tmp
    end
    if Vρ !== nothing
        Vρ .= tmp
    end
end
