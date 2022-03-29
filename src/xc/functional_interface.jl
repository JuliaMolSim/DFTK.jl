using ForwardDiff
import Libxc

#
# Abstract interface (TODO move to separate package, along with fallbacks)
#
abstract type Functional{F} end
family(::Functional{F})   where F = F
needs_σ(::Functional{F})  where F = (F in (:gga, :mgga, :mgga_lapl, ))
needs_τ(::Functional{F})  where F = (F in (      :mgga, :mgga_lapl, ))
needs_Δρ(::Functional{F}) where F = (F in (             :mgga_lapl, ))
has_energy(::Functional) = true  # Some functionals don't support energy evaluations
                                 # This is to inform the user that energy values will
                                 # not be returned.

# Silently drop extra arguments from evaluation functions
for fun in (:potential_terms, :kernel_terms)
    @eval begin
        $fun(func::Functional{:lda},  ρ, σ, args...)    = $fun(func, ρ)
        $fun(func::Functional{:gga},  ρ, σ, τ, args...) = $fun(func, ρ, σ)
        $fun(func::Functional{:mgga}, ρ, σ, τ, Δρ)      = $fun(func, ρ, σ, τ)
    end
end

#
# energy_per_particle functions
#
function energy_per_particle(::Val{identifier}, args...; kwargs...) where {identifier}
    error("Fallback functional for $identifier not implemented.")
end

include("lda_x.jl")
include("lda_c_vwn.jl")
include("lda_c_pw.jl")
include("gga_x_pbe.jl")
include("gga_c_pbe.jl")

#
# Fallback
#
struct FallbackFunctional{Family} <: Functional{Family}
    identifier::Symbol
end
Base.show(io::IO, fun::FallbackFunctional) = print(io, fun.identifier)
function potential_terms(func::FallbackFunctional{:lda}, ρ::AbstractArray)
    @assert ndims(ρ) == 4
    size(ρ, 1) == 1 || error("Fallback functionals only for one spin.")
    fE(ρ) = energy_per_particle(Val(func.identifier), ρ)
    fV(ρ) = ForwardDiff.derivative(ρ -> ρ*fE(ρ), ρ)
    (;zk=dropdims(fE.(ρ); dims=1), vrho=fV.(ρ))
end
function kernel_terms(func::FallbackFunctional{:lda}, ρ::AbstractArray)
    @assert ndims(ρ) == 4
    size(ρ, 1) == 1 || error("Fallback functionals only for one spin.")
    fE(ρ)  = energy_per_particle(Val(func.identifier), ρ)
    fV(ρ)  = ForwardDiff.derivative(ρ -> ρ*fE(ρ), ρ)
    fV2(ρ) = ForwardDiff.derivative(fV, ρ)
    (; zk=dropdims(fE.(ρ); dims=1), vrho=fV.(ρ), v2rho2=fV2.(ρ))
end
function potential_terms(func::FallbackFunctional{:gga}, ρ::AbstractArray, σ::AbstractArray)
    @assert ndims(ρ) == 4
    size(ρ, 1) == 1 || error("Fallback functionals only for one spin.")
    fE(ρ, σ)  = energy_per_particle(Val(func.identifier), ρ, σ)
    fVρ(ρ, σ) = ForwardDiff.derivative(ρ -> ρ*fE(ρ, σ), ρ)
    fVσ(ρ, σ) = ForwardDiff.derivative(σ -> ρ*fE(ρ, σ), σ)
    (;zk=dropdims(fE.(ρ, σ); dims=1), vrho=fVρ.(ρ, σ), vsigma=fVσ.(ρ, σ))
end

#
# Libxc (TODO Move this upstream, changing the interface of Libxc.jl)
#
struct LibxcFunctional{Family} <: Functional{Family}
    identifier::Symbol
end
Base.show(io::IO, fun::LibxcFunctional) = print(io, fun.identifier)
function LibxcFunctional(identifier::Symbol)
    fun = Libxc.Functional(identifier)
    @assert fun.family in (:lda, :gga, :mgga)
    if Libxc.needs_laplacian(fun)
        LibxcFunctional{:mgga_lapl}(identifier)
    else
        LibxcFunctional{fun.family}(identifier)
    end
end
function has_energy(func::LibxcFunctional)
    fun = Libxc.Functional(xc.identifier)
    0 in Libxc.supported_derivatives(fun)
end

function potential_terms(xc::LibxcFunctional{:lda}, ρ::AbstractArray{Float64})
    @assert ndims(ρ) == 4
    fun = Libxc.Functional(xc.identifier; n_spin=size(ρ, 1))
    derivatives = filter(i -> i in Libxc.supported_derivatives(fun), 0:1)
    Libxc.evaluate(fun; rho=ρ, derivatives)
end
function potential_terms(xc::LibxcFunctional{:gga}, ρ::AbstractArray{Float64},
                         σ::AbstractArray{Float64})
    @assert ndims(ρ) == 4
    fun = Libxc.Functional(xc.identifier; n_spin=size(ρ, 1))
    derivatives = filter(i -> i in Libxc.supported_derivatives(fun), 0:1)
    Libxc.evaluate(fun; rho=ρ, sigma=σ, derivatives)
end
function potential_terms(xc::LibxcFunctional{:mgga}, ρ::AbstractArray{Float64},
                         σ::AbstractArray{Float64}, τ::AbstractArray{Float64})
    @assert ndims(ρ) == 4
    fun = Libxc.Functional(xc.identifier; n_spin=size(ρ, 1))
    derivatives = filter(i -> i in Libxc.supported_derivatives(fun), 0:1)
    Libxc.evaluate(fun; rho=ρ, sigma=σ, tau=τ, derivatives)
end
function potential_terms(xc::LibxcFunctional{:mgga_lapl}, ρ::AbstractArray{Float64},
                         σ::AbstractArray{Float64}, τ::AbstractArray{Float64},
                         Δρ::AbstractArray{Float64})
    @assert ndims(ρ) == 4
    fun = Libxc.Functional(xc.identifier; n_spin=size(ρ, 1))
    derivatives = filter(i -> i in Libxc.supported_derivatives(fun), 0:1)
    Libxc.evaluate(fun; rho=ρ, sigma=σ, tau=τ, lapl=Δρ, derivatives)
end


function kernel_terms(xc::LibxcFunctional{:lda}, ρ::AbstractArray{Float64})
    @assert ndims(ρ) == 4
    fun = Libxc.Functional(xc.identifier; n_spin=size(ρ, 1))
    derivatives = filter(i -> i in Libxc.supported_derivatives(fun), 0:2)
    Libxc.evaluate(fun; rho=ρ, derivatives)
end
function kernel_terms(xc::LibxcFunctional{:gga}, ρ::AbstractArray{Float64},
                      σ::AbstractArray{Float64})
    @assert ndims(ρ) == 4
    fun = Libxc.Functional(xc.identifier; n_spin=size(ρ, 1))
    derivatives = filter(i -> i in Libxc.supported_derivatives(fun), 0:2)
    Libxc.evaluate(fun; rho=ρ, sigma=σ, derivatives)
end

#
# Automatically dispatching between both
# TODO Could be done by default for LibxcFunctionals ?
#      Could also be used to implement r-rules for LibxcFunctionals (via alternative primals)
#
struct DispatchFunctional{Family} <: Functional{Family}
    inner::LibxcFunctional{Family}
end
DispatchFunctional(identifier::Symbol) = DispatchFunctional(LibxcFunctional(identifier))
Base.show(io::IO, fun::DispatchFunctional) = print(io, fun.inner)
has_energy(func::DispatchFunctional) = has_energy(func.inner)

for fun in (:potential_terms, :kernel_terms)
    @eval begin
        function $fun(xc::DispatchFunctional, ρ::AbstractArray{Float64}, args...)
            $fun(xc.inner, ρ, args...)
        end
        function $fun(xc::DispatchFunctional{F}, ρ::AbstractArray, args...) where {F}
            fallback = FallbackFunctional{F}(xc.inner.identifier)
            $fun(fallback, ρ, args...)
        end
    end
end
