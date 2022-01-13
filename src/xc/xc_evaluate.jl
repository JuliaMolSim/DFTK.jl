import Libxc: evaluate!, Functional

include("lda_x.jl")
include("lda_c_vwn.jl")

# This file extends the evaluate! function from Libxc.jl for cases where the Array type
# is not a plain Julia array and the Floating point type is not Float64
function Libxc.evaluate!(func::Functional, ::Val{:lda}, ρ::AbstractArray;
                         zk=nothing, vrho=nothing, v2rho2=nothing)
    func.n_spin == 1  || error("Fallback functionals only for $(func.n_spin) == 1")

    zk = reshape(zk, size(ρ))
    if func.identifier == :lda_x
        fE = E_lda_x
    elseif func.identifier == :lda_c_vwn
        fE = E_lda_c_vwn
    end
    if zk !== nothing
        zk .= fE.(ρ)
    end

    fV(ρ) = ForwardDiff.derivative(ρ -> ρ*fE(ρ), ρ)
    if vrho !== nothing
        vrho .= fV.(ρ)
    end

    fV2(ρ) = ForwardDiff.derivative(fV)
    if v2rho2 !== nothing
        v2rho2 .= fV2.(ρ)
    end
end
