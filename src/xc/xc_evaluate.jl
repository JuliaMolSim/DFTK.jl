using ForwardDiff
import Libxc: evaluate!, Functional

function energy_per_particle(::Val{identifier}, args...; kwargs...) where {identifier}
    error("Fallback functional for $identifier not implemented.")
end

include("lda_x.jl")
include("lda_c_vwn.jl")

# Function that always dispatches to the DFTK fallback implementations
function xc_fallback!(func::Functional, ::Val{:lda}, ρ::AbstractArray;
                      zk=nothing, vrho=nothing, v2rho2=nothing)
    func.n_spin == 1  || error("Fallback functionals only for $(func.n_spin) == 1")
    zk = reshape(zk, size(ρ))

    fE = ρ -> energy_per_particle(Val(func.identifier), ρ)
    if zk !== nothing
        zk .= fE.(ρ)
    end

    fV(ρ) = ForwardDiff.derivative(ρ -> ρ*fE(ρ), ρ)
    if vrho !== nothing
        vrho .= fV.(ρ)
    end

    fV2(ρ) = ForwardDiff.derivative(fV, ρ)
    if v2rho2 !== nothing
        v2rho2 .= fV2.(ρ)
    end
end

# For cases where the Array type is not a plain Julia array and the Floating point
# type is not Float64, use xc_fallback! to evaluate the functional.
# Note: This is type piracy on the evaluate! function from Libxc.jl
Libxc.evaluate!(args...; kwargs...) = xc_fallback!(args...; kwargs...)
