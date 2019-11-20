using Libxc: evaluate_lda!, evaluate_gga!, Functional
using Libxc

include("lda_x.jl")
include("lda_c_vwn.jl")

# This file extends the evaluate_lda! from Libxc.jl for cases where the Array type
# is not a plain Julia array and the Floating point type is not Float64

function Libxc.evaluate_lda!(func::Functional, ρ::AbstractArray; E=nothing, Vρ=nothing)
    @assert func.family == Libxc.family_lda
    @assert func.n_spin == 1

    func.identifier == :lda_x     && return     lda_x!(ρ, E=E, Vρ=Vρ)
    func.identifier == :lda_c_vwn && return lda_c_vwn!(ρ, E=E, Vρ=Vρ)

    error("Fallback functional for $(string(func.identifier)) not implemented.")
end
