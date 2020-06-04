import Libxc: evaluate!, Functional

include("lda_x.jl")
include("lda_c_vwn.jl")

# This file extends the evaluate! function from Libxc.jl for cases where the Array type
# is not a plain Julia array and the Floating point type is not Float64
function Libxc.evaluate!(func::Functional, ::Val{:lda}, rho::AbstractArray;
                         zk=nothing, vrho=nothing, v2rho2=nothing)
    func.n_spin == 1  || error("Fallback functionals only for $n_spin == 1")
    isnothing(v2rho2) || error("Fallback functionals only for 0-th and 1-st derivative")

    func.identifier == :lda_x     && return     lda_x!(rho, E=zk, Vρ=vrho)
    func.identifier == :lda_c_vwn && return lda_c_vwn!(rho, E=zk, Vρ=vrho)

    error("Fallback functional for $(string(func.identifier)) not implemented.")
end
