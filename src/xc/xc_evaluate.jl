using ForwardDiff
import Libxc: evaluate!, Functional

function energy_per_particle(::Val{identifier}, args...; kwargs...) where {identifier}
    error("Fallback functional for $identifier not implemented.")
end

include("lda_x.jl")
include("lda_c_vwn.jl")
include("lda_c_pw.jl")
include("gga_x_pbe.jl")
include("gga_c_pbe.jl")

# Function that always dispatches to the DFTK fallback implementations
function xc_fallback!(func::Functional, ::Val{:lda}, ρ::AbstractArray;
                      zk=nothing, vrho=nothing, v2rho2=nothing)
    func.n_spin == 1  || error("Fallback functionals only for $(func.n_spin) == 1")
    zk = reshape(zk, size(ρ))

    fE(ρ) = energy_per_particle(Val(func.identifier), ρ)
    if !isnothing(zk)
        zk .= fE.(ρ)
    end

    fV(ρ) = ForwardDiff.derivative(ρ -> ρ*fE(ρ), ρ)
    if !isnothing(vrho)
        vrho .= fV.(ρ)
    end

    fV2(ρ) = ForwardDiff.derivative(fV, ρ)
    if !isnothing(v2rho2)
        v2rho2 .= fV2.(ρ)
    end
end
function xc_fallback!(func::Functional, ::Val{:gga}, ρ::AbstractArray;
                      sigma, zk=nothing, vrho=nothing, vsigma=nothing,
                      v2rho2=nothing, v2rhosigma=nothing, v2sigma2=nothing)
    func.n_spin == 1  || error("Fallback functionals only for $(func.n_spin) == 1")
    zk = reshape(zk, size(ρ))
    σ  = sigma

    fE(ρ, σ) = energy_per_particle(Val(func.identifier), ρ, σ)
    if zk !== nothing
        zk .= fE.(ρ, σ)
    end

    fVρ(ρ, σ) = ForwardDiff.derivative(ρ -> ρ*fE(ρ, σ), ρ)
    fVσ(ρ, σ) = ForwardDiff.derivative(σ -> ρ*fE(ρ, σ), σ)
    if !isnothing(vrho) && !isnothing(vsigma)
        vrho   .= fVρ.(ρ, σ)
        vsigma .= fVσ.(ρ, σ)
    end

    @assert isnothing(v2rho2)
    @assert isnothing(v2rhosigma)
    @assert isnothing(v2sigma2)
end

# For cases where the Array type is not a plain Julia array and the Floating point
# type is not Float64, use xc_fallback! to evaluate the functional.
# Note: This is type piracy on the evaluate! function from Libxc.jl
Libxc.evaluate!(args...; kwargs...) = xc_fallback!(args...; kwargs...)
