using Libxc: evaluate_lda!, Functional
using Libxc

include("xc_fallback/lda_x.jl")
include("xc_fallback/lda_c_vwn.jl")

# This file extends the evaluate_lda! from Libxc.jl for cases where the Array type
# is not a plain Julia array and the Floating point type is not Float64

function Libxc.evaluate_lda!(func::Functional, ρ::AbstractArray; E=nothing, Vρ=nothing)
    @assert func.family == Libxc.family_lda
    @assert func.n_spin == 1

    if func.name == "lda_x"
        return lda_x!(ρ, E=E, Vρ=Vρ)
    elseif func.name == "lda_c_vwn"
        return lda_c_vwn!(ρ, E=E, Vρ=Vρ)
    else
        error("Fallback functional for name $(func.name) not implemented.")
    end
end
