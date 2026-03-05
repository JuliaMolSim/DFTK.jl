using DftFunctionals
import ForwardDiff: Dual
import Libxc

#
# Libxc (TODO Move this upstream, changing the interface of Libxc.jl)
#
struct LibxcFunctional{Family,Kind} <: Functional{Family,Kind}
    identifier::Symbol
end
function LibxcFunctional(identifier::Symbol)
    fun = Libxc.Functional(identifier)
    @assert fun.kind   in (:exchange, :correlation, :exchange_correlation)
    kind = Dict(:exchange => :x, :correlation => :c, :exchange_correlation => :xc)[fun.kind]

    @assert fun.family in (:lda, :gga, :mgga, :hyb_lda, :hyb_gga, :hyb_mgga)
    # Libxc maintains the distinction between hybrid and non-hybrid equivalents,
    # but this distinction is not relevant for the functional interface
    if startswith(string(fun.family), "hyb")
        family = Symbol(string(fun.family)[5:end])
    else
        family = fun.family
    end
    if family == :mgga && Libxc.needs_laplacian(fun)
        family = :mggal
    end
    LibxcFunctional{family,kind}(identifier)
end

DftFunctionals.identifier(fun::LibxcFunctional) = fun.identifier
function DftFunctionals.has_energy(func::LibxcFunctional)
    0 in Libxc.supported_derivatives(Libxc.Functional(func.identifier))
end

function libxc_energy(terms, ρ)
    haskey(terms, :zk) ? reshape(terms.zk, 1, size(ρ, 2)) .* sum(ρ; dims=1) : false
end

function DftFunctionals.potential_terms(func::LibxcFunctional{:lda}, ρ::AbstractMatrix{Float64})
    s_ρ, n_p = size(ρ)
    fun = Libxc.Functional(func.identifier; n_spin=s_ρ)
    derivatives = filter(in(Libxc.supported_derivatives(fun)), 0:1)
    terms = Libxc.evaluate(fun; rho=ρ, derivatives)
    e  = libxc_energy(terms, ρ)
    Vρ = reshape(terms.vrho, s_ρ, n_p)
    (; e, Vρ)
end
function DftFunctionals.potential_terms(func::LibxcFunctional{:gga}, ρ::AbstractMatrix{Float64},
                                        σ::AbstractMatrix{Float64})
    s_ρ, n_p = size(ρ)
    s_σ = size(σ, 1)
    fun = Libxc.Functional(func.identifier; n_spin=s_ρ)
    derivatives = filter(in(Libxc.supported_derivatives(fun)), 0:1)
    terms = Libxc.evaluate(fun; rho=ρ, sigma=σ, derivatives)
    e  = libxc_energy(terms, ρ)
    Vρ = reshape(terms.vrho,   s_ρ, n_p)
    Vσ = reshape(terms.vsigma, s_σ, n_p)
    (; e, Vρ, Vσ)
end
function DftFunctionals.potential_terms(func::LibxcFunctional{:mgga}, ρ::AbstractMatrix{Float64},
                                        σ::AbstractMatrix{Float64}, τ::AbstractMatrix{Float64})
    s_ρ, n_p = size(ρ)
    s_σ = size(σ, 1)
    fun = Libxc.Functional(func.identifier; n_spin=s_ρ)
    derivatives = filter(in(Libxc.supported_derivatives(fun)), 0:1)
    terms = Libxc.evaluate(fun; rho=ρ, sigma=σ, tau=τ, derivatives)
    e  = libxc_energy(terms, ρ)
    Vρ = reshape(terms.vrho,   s_ρ, n_p)
    Vσ = reshape(terms.vsigma, s_σ, n_p)
    Vτ = reshape(terms.vtau,   s_ρ, n_p)
    (; e, Vρ, Vσ, Vτ)
end
function DftFunctionals.potential_terms(func::LibxcFunctional{:mggal}, ρ::AbstractMatrix{Float64},
                                        σ::AbstractMatrix{Float64}, τ::AbstractMatrix{Float64},
                                        Δρ::AbstractMatrix{Float64})
    s_ρ, n_p = size(ρ)
    s_σ = size(σ, 1)
    fun = Libxc.Functional(func.identifier; n_spin=s_ρ)
    derivatives = filter(in(Libxc.supported_derivatives(fun)), 0:1)
    terms = Libxc.evaluate(fun; rho=ρ, sigma=σ, tau=τ, lapl=Δρ, derivatives)
    e  = libxc_energy(terms, ρ)
    Vρ = reshape(terms.vrho,   s_ρ, n_p)
    Vσ = reshape(terms.vsigma, s_σ, n_p)
    Vτ = reshape(terms.vtau,   s_ρ, n_p)
    Vl = reshape(terms.vlapl,  s_ρ, n_p)
    (; e, Vρ, Vσ, Vτ, Vl)
end

# Kernel support via automatic differentiation
#
# We invoke Libxc.evaluate at the same point, but ask for one more derivative.
# Then we manually multiply the second derivatives by the given perturbation (δρ etc)
# to compute δVs to return.
# For collinear spins:
# - ρ has s_ρ == 2 components and σ has s_σ == 3 components.
# - There are cross-spin-component derivatives which we sum up manually.
#   For example in LDA the change in Vρ₁ is ∂²E_xc/∂ρ₁² * δρ₁ + ∂²E_xc/∂ρ₁∂ρ₂ * δρ₂,
#   and similarly for Vρ₂.
#   For GGA, there are also cross-derivatives with σ, e.g. ∂²E_xc/∂ρ₁∂σ₁ * δσ₁, etc.
#   This is handled by the various libxc_assemble_δV functions below.
# - libxc returns the cross-spin derivatives in a compact form,
#   see https://libxc.gitlab.io/manual/libxc-5.1.x/

# Combine N vectors of size (n_p) into one (N, n_p) array
libxc_combine_spins(xs...) = reduce(vcat, transpose.(xs))

# Helper functions to compute δVρ, δVσ, δVτ from the second derivatives
# and the perturbations, by summing up all the spin combinations.
@views function libxc_assemble_δVρ(Vρρ, δρ, Vρσ=nothing, δσ=nothing,
                                            Vρτ=nothing, δτ=nothing,
                                            Vρl=nothing, δl=nothing)
    if size(δρ, 1) == 1
        δVρ = Vρρ .* δρ
        isnothing(Vρσ) || (δVρ .+= Vρσ .* δσ)
        isnothing(Vρτ) || (δVρ .+= Vρτ .* δτ)
        isnothing(Vρl) || (δVρ .+= Vρl .* δl)
        return δVρ
    else
        δVρ1 = @. Vρρ[1,:] * δρ[1,:] + Vρρ[2,:] * δρ[2,:]
        δVρ2 = @. Vρρ[2,:] * δρ[1,:] + Vρρ[3,:] * δρ[2,:]
        if !isnothing(Vρσ)
            δVρ1 .+= @. Vρσ[1,:] * δσ[1,:] + Vρσ[2,:] * δσ[2,:] + Vρσ[3,:] * δσ[3,:]
            δVρ2 .+= @. Vρσ[4,:] * δσ[1,:] + Vρσ[5,:] * δσ[2,:] + Vρσ[6,:] * δσ[3,:]
        end
        if !isnothing(Vρτ)
            δVρ1 .+= @. Vρτ[1,:] * δτ[1,:] + Vρτ[2,:] * δτ[2,:]
            δVρ2 .+= @. Vρτ[3,:] * δτ[1,:] + Vρτ[4,:] * δτ[2,:]
        end
        if !isnothing(Vρl)
            δVρ1 .+= @. Vρl[1,:] * δl[1,:] + Vρl[2,:] * δl[2,:]
            δVρ2 .+= @. Vρl[3,:] * δl[1,:] + Vρl[4,:] * δl[2,:]
        end
        return libxc_combine_spins(δVρ1, δVρ2)
    end
end
@views function libxc_assemble_δVσ(Vρσ, δρ, Vσσ, δσ, Vστ=nothing, δτ=nothing,
                                                     Vσl=nothing, δl=nothing)
    if size(δρ, 1) == 1
        δVσ = Vρσ .* δρ .+ Vσσ .* δσ
        isnothing(Vστ) || (δVσ .+= Vστ .* δτ)
        isnothing(Vσl) || (δVσ .+= Vσl .* δl)
        return δVσ
    else
        δVσ1 =   @. Vρσ[1,:] * δρ[1,:] + Vρσ[4,:] * δρ[2,:]
        δVσ2 =   @. Vρσ[2,:] * δρ[1,:] + Vρσ[5,:] * δρ[2,:]
        δVσ3 =   @. Vρσ[3,:] * δρ[1,:] + Vρσ[6,:] * δρ[2,:]
        δVσ1 .+= @. Vσσ[1,:] * δσ[1,:] + Vσσ[2,:] * δσ[2,:] + Vσσ[3,:] * δσ[3,:]
        δVσ2 .+= @. Vσσ[2,:] * δσ[1,:] + Vσσ[4,:] * δσ[2,:] + Vσσ[5,:] * δσ[3,:]
        δVσ3 .+= @. Vσσ[3,:] * δσ[1,:] + Vσσ[5,:] * δσ[2,:] + Vσσ[6,:] * δσ[3,:]
        if !isnothing(Vστ)
            δVσ1 .+= @. Vστ[1,:]*δτ[1,:] + Vστ[2,:]*δτ[2,:]
            δVσ2 .+= @. Vστ[3,:]*δτ[1,:] + Vστ[4,:]*δτ[2,:]
            δVσ3 .+= @. Vστ[5,:]*δτ[1,:] + Vστ[6,:]*δτ[2,:]
        end
        if !isnothing(Vσl)
            δVσ1 .+= @. Vσl[1,:]*δl[1,:] + Vσl[2,:]*δl[2,:]
            δVσ2 .+= @. Vσl[3,:]*δl[1,:] + Vσl[4,:]*δl[2,:]
            δVσ3 .+= @. Vσl[5,:]*δl[1,:] + Vσl[6,:]*δl[2,:]
        end
        return libxc_combine_spins(δVσ1, δVσ2, δVσ3)
    end
end
@views function libxc_assemble_δVτ(Vρτ, δρ, Vστ, δσ, Vττ, δτ,
                                   Vlτ=nothing, δl=nothing)
    if size(δρ, 1) == 1
        δVτ = Vρτ .* δρ .+ Vστ .* δσ .+ Vττ .* δτ
        isnothing(Vlτ) || (δVτ .+= Vlτ .* δl)
        return δVτ
    else
        δVτ1 =   @. Vρτ[1,:] * δρ[1,:] + Vρτ[3,:] * δρ[2,:]
        δVτ2 =   @. Vρτ[2,:] * δρ[1,:] + Vρτ[4,:] * δρ[2,:]
        δVτ1 .+= @. Vστ[1,:] * δσ[1,:] + Vστ[3,:] * δσ[2,:] + Vστ[5,:] * δσ[3,:]
        δVτ2 .+= @. Vστ[2,:] * δσ[1,:] + Vστ[4,:] * δσ[2,:] + Vστ[6,:] * δσ[3,:]
        δVτ1 .+= @. Vττ[1,:] * δτ[1,:] + Vττ[2,:] * δτ[2,:]
        δVτ2 .+= @. Vττ[2,:] * δτ[1,:] + Vττ[3,:] * δτ[2,:]
        if !isnothing(Vlτ)
            δVτ1 .+= @. Vlτ[1,:]*δl[1,:] + Vlτ[3,:]*δl[2,:]
            δVτ2 .+= @. Vlτ[2,:]*δl[1,:] + Vlτ[4,:]*δl[2,:]
        end
        return libxc_combine_spins(δVτ1, δVτ2)
    end
end
@views function libxc_assemble_δVl(Vρl, δρ, Vσl, δσ, Vlτ, δτ, Vll, δl)
    if size(δρ, 1) == 1
        return Vρl .* δρ .+ Vσl .* δσ .+ Vlτ .* δτ .+ Vll .* δl
    else
        δVl1 =   @. Vρl[1,:] * δρ[1,:] + Vρl[3,:] * δρ[2,:]
        δVl2 =   @. Vρl[2,:] * δρ[1,:] + Vρl[4,:] * δρ[2,:]
        δVl1 .+= @. Vσl[1,:] * δσ[1,:] + Vσl[3,:] * δσ[2,:] + Vσl[5,:] * δσ[3,:]
        δVl2 .+= @. Vσl[2,:] * δσ[1,:] + Vσl[4,:] * δσ[2,:] + Vσl[6,:] * δσ[3,:]
        δVl1 .+= @. Vlτ[1,:] * δτ[1,:] + Vlτ[2,:] * δτ[2,:]
        δVl2 .+= @. Vlτ[3,:] * δτ[1,:] + Vlτ[4,:] * δτ[2,:]
        δVl1 .+= @. Vll[1,:] * δl[1,:] + Vll[2,:] * δl[2,:]
        δVl2 .+= @. Vll[2,:] * δl[1,:] + Vll[3,:] * δl[2,:]
        return libxc_combine_spins(δVl1, δVl2)
    end
end

@views function DftFunctionals.potential_terms(func::LibxcFunctional{:lda},
                                               ρ_δρ::AbstractMatrix{DT}
                                               ) where {N,T,DT<:Dual{T,Float64,N}}
    ρ = ForwardDiff.value.(ρ_δρ)
    s_ρ, n_p = size(ρ)
    fun = Libxc.Functional(func.identifier; n_spin=s_ρ)
    derivatives = filter(in(Libxc.supported_derivatives(fun)), 0:2)
    terms = Libxc.evaluate(fun; rho=ρ, derivatives)
    e = libxc_energy(terms, ρ)
    Vρ = reshape(terms.vrho, s_ρ, n_p)
    Vρρ = terms.v2rho2

    δe = ntuple(Val(N)) do n
        sum(Vρ .* ForwardDiff.partials.(ρ_δρ, n); dims=1)
    end
    δVρ = ntuple(Val(N)) do n
        libxc_assemble_δVρ(Vρρ, ForwardDiff.partials.(ρ_δρ, n))
    end
    (; e=map(Dual{T}, e, δe...),
       Vρ=map(Dual{T}, Vρ, δVρ...))
end
@views function DftFunctionals.potential_terms(func::LibxcFunctional{:gga},
                                               ρ_δρ::AbstractMatrix{DT},
                                               σ_δσ::AbstractMatrix{DT}
                                               ) where {N,T,DT<:Dual{T,Float64,N}}
    ρ = ForwardDiff.value.(ρ_δρ)
    σ = ForwardDiff.value.(σ_δσ)
    s_ρ, n_p = size(ρ)
    s_σ = size(σ, 1)
    fun = Libxc.Functional(func.identifier; n_spin=s_ρ)
    derivatives = filter(in(Libxc.supported_derivatives(fun)), 0:2)
    terms = Libxc.evaluate(fun; rho=ρ, sigma=σ, derivatives)
    e  = libxc_energy(terms, ρ)
    Vρ = reshape(terms.vrho,   s_ρ, n_p)
    Vσ = reshape(terms.vsigma, s_σ, n_p)
    Vρρ = terms.v2rho2
    Vρσ = terms.v2rhosigma
    Vσσ = terms.v2sigma2

    δe = ntuple(Val(N)) do n
        ( sum(Vρ .* ForwardDiff.partials.(ρ_δρ, n); dims=1)
        + sum(Vσ .* ForwardDiff.partials.(σ_δσ, n); dims=1))
    end
    δVρ = ntuple(Val(N)) do n
        libxc_assemble_δVρ(Vρρ, ForwardDiff.partials.(ρ_δρ, n),
                           Vρσ, ForwardDiff.partials.(σ_δσ, n))
    end
    δVσ = ntuple(Val(N)) do n
        libxc_assemble_δVσ(Vρσ, ForwardDiff.partials.(ρ_δρ, n),
                           Vσσ, ForwardDiff.partials.(σ_δσ, n))
    end
    (; e=map(Dual{T},   e, δe...),
       Vρ=map(Dual{T}, Vρ, δVρ...),
       Vσ=map(Dual{T}, Vσ, δVσ...))
end
@views function DftFunctionals.potential_terms(func::LibxcFunctional{:mgga},
                                               ρ_δρ::AbstractMatrix{DT},
                                               σ_δσ::AbstractMatrix{DT},
                                               τ_δτ::AbstractMatrix{DT}
                                               ) where {N,T,DT<:Dual{T,Float64,N}}
    ρ = ForwardDiff.value.(ρ_δρ)
    σ = ForwardDiff.value.(σ_δσ)
    τ = ForwardDiff.value.(τ_δτ)
    s_ρ, n_p = size(ρ)
    s_σ = size(σ, 1)
    fun = Libxc.Functional(func.identifier; n_spin=s_ρ)
    derivatives = filter(in(Libxc.supported_derivatives(fun)), 0:2)
    terms = Libxc.evaluate(fun; rho=ρ, sigma=σ, tau=τ, derivatives)
    e  = libxc_energy(terms, ρ)
    Vρ = reshape(terms.vrho,   s_ρ, n_p)
    Vσ = reshape(terms.vsigma, s_σ, n_p)
    Vτ = reshape(terms.vtau,   s_ρ, n_p)
    Vρρ = terms.v2rho2
    Vρσ = terms.v2rhosigma
    Vρτ = terms.v2rhotau
    Vσσ = terms.v2sigma2
    Vστ = terms.v2sigmatau
    Vττ = terms.v2tau2

    δe = ntuple(Val(N)) do n
        ( sum(Vρ .* ForwardDiff.partials.(ρ_δρ, n); dims=1)
        + sum(Vσ .* ForwardDiff.partials.(σ_δσ, n); dims=1)
        + sum(Vτ .* ForwardDiff.partials.(τ_δτ, n); dims=1))
    end
    δVρ = ntuple(Val(N)) do n
        libxc_assemble_δVρ(Vρρ, ForwardDiff.partials.(ρ_δρ, n),
                           Vρσ, ForwardDiff.partials.(σ_δσ, n),
                           Vρτ, ForwardDiff.partials.(τ_δτ, n))
    end
    δVσ = ntuple(Val(N)) do n
        libxc_assemble_δVσ(Vρσ, ForwardDiff.partials.(ρ_δρ, n),
                           Vσσ, ForwardDiff.partials.(σ_δσ, n),
                           Vστ, ForwardDiff.partials.(τ_δτ, n))
    end
    δVτ = ntuple(Val(N)) do n
        libxc_assemble_δVτ(Vρτ, ForwardDiff.partials.(ρ_δρ, n),
                           Vστ, ForwardDiff.partials.(σ_δσ, n),
                           Vττ, ForwardDiff.partials.(τ_δτ, n))
    end
    (; e=map(Dual{T},   e, δe...),
       Vρ=map(Dual{T}, Vρ, δVρ...),
       Vσ=map(Dual{T}, Vσ, δVσ...),
       Vτ=map(Dual{T}, Vτ, δVτ...))
end
@views function DftFunctionals.potential_terms(func::LibxcFunctional{:mggal},
                                               ρ_δρ::AbstractMatrix{DT},
                                               σ_δσ::AbstractMatrix{DT},
                                               τ_δτ::AbstractMatrix{DT},
                                               l_δl::AbstractMatrix{DT}
                                               ) where {N,T,DT<:Dual{T,Float64,N}}
    ρ = ForwardDiff.value.(ρ_δρ)
    σ = ForwardDiff.value.(σ_δσ)
    τ = ForwardDiff.value.(τ_δτ)
    l = ForwardDiff.value.(l_δl)
    s_ρ, n_p = size(ρ)
    s_σ = size(σ, 1)
    fun = Libxc.Functional(func.identifier; n_spin=s_ρ)
    derivatives = filter(in(Libxc.supported_derivatives(fun)), 0:2)
    terms = Libxc.evaluate(fun; rho=ρ, sigma=σ, tau=τ, lapl=l, derivatives)
    e  = libxc_energy(terms, ρ)
    Vρ = reshape(terms.vrho,   s_ρ, n_p)
    Vσ = reshape(terms.vsigma, s_σ, n_p)
    Vτ = reshape(terms.vtau,   s_ρ, n_p)
    Vl = reshape(terms.vlapl,  s_ρ, n_p)
    Vρρ = terms.v2rho2
    Vρσ = terms.v2rhosigma
    Vρτ = terms.v2rhotau
    Vρl = terms.v2rholapl
    Vσσ = terms.v2sigma2
    Vστ = terms.v2sigmatau
    Vσl = terms.v2sigmalapl
    Vll = terms.v2lapl2
    Vlτ = terms.v2lapltau
    Vττ = terms.v2tau2

    δe = ntuple(Val(N)) do n
        ( sum(Vρ .* ForwardDiff.partials.(ρ_δρ, n); dims=1)
        + sum(Vσ .* ForwardDiff.partials.(σ_δσ, n); dims=1)
        + sum(Vτ .* ForwardDiff.partials.(τ_δτ, n); dims=1)
        + sum(Vl .* ForwardDiff.partials.(l_δl, n); dims=1))
    end
    δVρ = ntuple(Val(N)) do n
        libxc_assemble_δVρ(Vρρ, ForwardDiff.partials.(ρ_δρ, n),
                           Vρσ, ForwardDiff.partials.(σ_δσ, n),
                           Vρτ, ForwardDiff.partials.(τ_δτ, n),
                           Vρl, ForwardDiff.partials.(l_δl, n))
    end
    δVσ = ntuple(Val(N)) do n
        libxc_assemble_δVσ(Vρσ, ForwardDiff.partials.(ρ_δρ, n),
                           Vσσ, ForwardDiff.partials.(σ_δσ, n),
                           Vστ, ForwardDiff.partials.(τ_δτ, n),
                           Vσl, ForwardDiff.partials.(l_δl, n))
    end
    δVτ = ntuple(Val(N)) do n
        libxc_assemble_δVτ(Vρτ, ForwardDiff.partials.(ρ_δρ, n),
                           Vστ, ForwardDiff.partials.(σ_δσ, n),
                           Vττ, ForwardDiff.partials.(τ_δτ, n),
                           Vlτ, ForwardDiff.partials.(l_δl, n))
    end
    δVl = ntuple(Val(N)) do n
        libxc_assemble_δVl(Vρl, ForwardDiff.partials.(ρ_δρ, n),
                           Vσl, ForwardDiff.partials.(σ_δσ, n),
                           Vlτ, ForwardDiff.partials.(τ_δτ, n),
                           Vll, ForwardDiff.partials.(l_δl, n))
    end
    (; e=map(Dual{T},   e, δe...),
       Vρ=map(Dual{T}, Vρ, δVρ...),
       Vσ=map(Dual{T}, Vσ, δVσ...),
       Vτ=map(Dual{T}, Vτ, δVτ...),
       Vl=map(Dual{T}, Vl, δVl...))
end

#
# Automatic dispatching between Libxc (where possible) and the generic implementation
# in DftFunctionals (where needed).
# TODO Could be done by default for LibxcFunctionals ?
#      Could also be used to implement r-rules for LibxcFunctionals (via alternative primals)
#      Could also be moved into a package on its own?
struct DispatchFunctional{Family,Kind} <: Functional{Family,Kind}
    inner::LibxcFunctional{Family,Kind}
end
DispatchFunctional(identifier::Symbol) = DispatchFunctional(LibxcFunctional(identifier))
DftFunctionals.identifier(fun::DispatchFunctional) = identifier(fun.inner)
DftFunctionals.has_energy(fun::DispatchFunctional) = has_energy(fun.inner)

# Note: CuMatrix dispatch to Libxc.jl is defined in src/workarounds/cuda_arrays.jl
const DispatchFloat = Union{Float64,Dual{<:Any,Float64}}
function DftFunctionals.potential_terms(fun::DispatchFunctional, ρ::Matrix{<:DispatchFloat}, args...)
    potential_terms(fun.inner, ρ, args...)
end
function DftFunctionals.potential_terms(fun::DispatchFunctional, ρ::AbstractMatrix, args...)
    potential_terms(DftFunctional(identifier(fun)), ρ, args...)
end

# TODO This is hackish for now until Libxc has fully picked up the DftFunctionals.jl interface
exx_coefficient(::Functional{:lda})      = nothing
exx_coefficient(::Functional{:gga})      = nothing
exx_coefficient(::Functional{:mgga})     = nothing
exx_coefficient(fun::DispatchFunctional) = exx_coefficient(fun.inner)
exx_coefficient(fun::LibxcFunctional)    = Libxc.Functional(fun.identifier).exx_coefficient
