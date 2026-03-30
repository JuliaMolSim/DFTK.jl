using DftFunctionals
import ForwardDiff: Dual
import Libxc

# TODO: Observations for a future XC functional interface refactor:
#   - The distinction between mgga and mggal makes no sense as some mggal don't have a τ
#     Merge them and make both τ and lapl optional
#   - No need for Dftfunctionals.kernel_terms This can be strictly expressed by AD.
#     In contrast we no need both energy_density and potential_terms as some functionals
#     don't have an energy.

#
# Libxc (TODO Move this upstream, changing the interface of Libxc.jl)
#
struct LibxcFunctional{Family,Kind} <: Functional{Family,Kind}
    identifier::Symbol
    has_energy::Bool
    needs_tau::Bool
    needs_laplacian::Bool
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
    LibxcFunctional{family,kind}(identifier,
                                 0 in Libxc.supported_derivatives(fun),  # has_energy
                                 Libxc.needs_tau(fun),
                                 Libxc.needs_laplacian(fun))
end
DftFunctionals.identifier(fun::LibxcFunctional)  = fun.identifier
DftFunctionals.has_energy(func::LibxcFunctional) = func.has_energy
DftFunctionals.needs_τ(fun::LibxcFunctional)     = fun.needs_tau
DftFunctionals.needs_Δρ(fun::LibxcFunctional)    = fun.needs_laplacian

function libxc_energy_density(terms::NamedTuple, ρ::AbstractMatrix)
    haskey(terms, :zk) ? reshape(terms.zk, 1, size(ρ, 2)) .* sum(ρ; dims=1) : false
end

function DftFunctionals.energy_density(func::LibxcFunctional, ρ::AbstractMatrix{Float64},
                                       σ=nothing, τ=nothing, Δρ=nothing)
    terms = (; )
    if has_energy(func)
        libxcfun = Libxc.Functional(func.identifier; n_spin=size(ρ, 1))
        terms = Libxc.evaluate(libxcfun; derivatives=0:0, rho=ρ, sigma=σ, tau=τ, lapl=Δρ)
    end
    libxc_energy_density(terms, ρ)
end

#
# AD support for energy density
#
# Helper functions to compute δe, from the first derivatives
# and the perturbations, by summing up all the spin combinations.
@views function libxc_assemble_δe(Vρ, δρ, Vσ=nothing, δσ=nothing,
                                          Vτ=nothing, δτ=nothing,
                                          Vl=nothing, δl=nothing)
    δe = sum(Vρ .* δρ; dims=1)
    isnothing(Vσ) || (δe += sum(Vσ .* δσ; dims=1))
    isnothing(Vτ) || (δe += sum(Vτ .* δτ; dims=1))
    isnothing(Vl) || (δe += sum(Vl .* δl; dims=1))
    δe
end

get_partials_(x_δx::AbstractArray, n::Integer) = ForwardDiff.partials.(x_δx, n)
get_partials_(::Nothing, ::Integer)            = nothing

function DftFunctionals.energy_density(func::LibxcFunctional{:lda},
                                       ρ_δρ::AbstractMatrix{DT}, args...
                                       ) where {N,T,Tg,DT<:Dual{Tg,T,N}}
    has_energy(func) || return zero(T)
    ρ = ForwardDiff.value.(ρ_δρ)
    (; e, Vρ) = potential_terms(func, ρ)
    δe = ntuple(Val(N)) do n
        libxc_assemble_δe(Vρ, get_partials_(ρ_δρ, n))
    end
    map(Dual{Tg}, e, δe...)
end
function DftFunctionals.energy_density(func::LibxcFunctional{:gga}, ρ_δρ::AbstractMatrix{DT},
                        σ_δσ::AbstractMatrix{DT}, args...
                        ) where {N,T,Tg,DT<:Dual{Tg,T,N}}
    has_energy(func) || return zero(T)
    ρ = ForwardDiff.value.(ρ_δρ)
    σ = ForwardDiff.value.(σ_δσ)
    (; e, Vρ, Vσ) = potential_terms(func, ρ, σ)
    δe = ntuple(Val(N)) do n
        libxc_assemble_δe(Vρ, get_partials_(ρ_δρ, n),
                          Vσ, get_partials_(σ_δσ, n))
    end
    map(Dual{Tg}, e, δe...)
end
function DftFunctionals.energy_density(func::LibxcFunctional{:mgga}, ρ_δρ::AbstractMatrix{DT},
                                       σ_δσ::AbstractMatrix{DT}, τ_δτ::AbstractMatrix{DT}, args...
                                       ) where {N,T,Tg,DT<:Dual{Tg,T,N}}
    has_energy(func) || return zero(T)
    ρ = ForwardDiff.value.(ρ_δρ)
    σ = ForwardDiff.value.(σ_δσ)
    τ = ForwardDiff.value.(τ_δτ)
    (; e, Vρ, Vσ, Vτ) = potential_terms(func, ρ, σ, τ)
    δe = ntuple(Val(N)) do n
        libxc_assemble_δe(Vρ, get_partials_(ρ_δρ, n),
                          Vσ, get_partials_(σ_δσ, n),
                          Vτ, get_partials_(τ_δτ, n))
    end
    map(Dual{Tg}, e, δe...)
end
function DftFunctionals.energy_density(func::LibxcFunctional{:mggal},
                                       ρ_δρ::AbstractMatrix{DT},
                                       σ_δσ::AbstractMatrix{DT},
                                       τ_δτ::Union{Nothing,AbstractMatrix{DT}},
                                       l_δl::AbstractMatrix{DT}
                                       ) where {N,T,Tg,DT<:Dual{Tg,T,N}}
    has_energy(func) || return zero(T)
    ρ = ForwardDiff.value.(ρ_δρ)
    σ = ForwardDiff.value.(σ_δσ)
    τ = ForwardDiff.value.(τ_δτ)
    lapl = ForwardDiff.value.(l_δl)
    terms = potential_terms(func, ρ, σ, τ, lapl)

    δe = ntuple(Val(N)) do n
        Vτ = get(terms, :Vτ, nothing)
        libxc_assemble_δe(terms.Vρ, get_partials_(ρ_δρ, n),
                          terms.Vσ, get_partials_(σ_δσ, n),
                                Vτ, get_partials_(τ_δτ, n),
                          terms.Vl, get_partials_(l_δl, n))
    end
    map(Dual{Tg}, terms.e, δe...)
end

#
# Potential terms
#
function DftFunctionals.potential_terms(func::LibxcFunctional{:lda}, ρ::AbstractMatrix{Float64})
    s_ρ, n_p = size(ρ)
    fun = Libxc.Functional(func.identifier; n_spin=s_ρ)
    derivatives = filter(in(Libxc.supported_derivatives(fun)), 0:1)
    terms = Libxc.evaluate(fun; rho=ρ, derivatives)
    e  = libxc_energy_density(terms, ρ)
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
    e  = libxc_energy_density(terms, ρ)
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
    e  = libxc_energy_density(terms, ρ)
    Vρ = reshape(terms.vrho,   s_ρ, n_p)
    Vσ = reshape(terms.vsigma, s_σ, n_p)
    Vτ = reshape(terms.vtau,   s_ρ, n_p)
    (; e, Vρ, Vσ, Vτ)
end
function DftFunctionals.potential_terms(func::LibxcFunctional{:mggal},
                                        ρ::AbstractMatrix{Float64},
                                        σ::AbstractMatrix{Float64},
                                        τ::Union{Nothing,AbstractMatrix{Float64}},
                                        Δρ::AbstractMatrix{Float64})
    s_ρ, n_p = size(ρ)
    s_σ = size(σ, 1)
    fun = Libxc.Functional(func.identifier; n_spin=s_ρ)
    derivatives = filter(in(Libxc.supported_derivatives(fun)), 0:1)
    terms = Libxc.evaluate(fun; rho=ρ, sigma=σ, tau=τ, lapl=Δρ, derivatives)
    e  = libxc_energy_density(terms, ρ)
    Vρ = reshape(terms.vrho,   s_ρ, n_p)
    Vσ = reshape(terms.vsigma, s_σ, n_p)
    Vl = reshape(terms.vlapl, s_ρ, n_p)

    if haskey(terms, :vtau)
        Vτ = reshape(terms.vtau, s_ρ, n_p)
        (; e, Vρ, Vσ, Vτ, Vl)
    else
        (; e, Vρ, Vσ, Vl)
    end
end

#
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
        !isnothing(Vlτ) && (δVτ .+= Vlτ .* δl)
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
        δVl = Vρl .* δρ .+ Vσl .* δσ .+ Vll .* δl
        !isnothing(Vlτ) && (δVl .+= Vlτ .* δτ)
        return δVl
    else
        δVl1 =   @. Vρl[1,:] * δρ[1,:] + Vρl[3,:] * δρ[2,:]
        δVl2 =   @. Vρl[2,:] * δρ[1,:] + Vρl[4,:] * δρ[2,:]
        δVl1 .+= @. Vσl[1,:] * δσ[1,:] + Vσl[3,:] * δσ[2,:] + Vσl[5,:] * δσ[3,:]
        δVl2 .+= @. Vσl[2,:] * δσ[1,:] + Vσl[4,:] * δσ[2,:] + Vσl[6,:] * δσ[3,:]
        if !isnothing(Vlτ)
            δVl1 .+= @. Vlτ[1,:] * δτ[1,:] + Vlτ[2,:] * δτ[2,:]
            δVl2 .+= @. Vlτ[3,:] * δτ[1,:] + Vlτ[4,:] * δτ[2,:]
        end
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
    e = libxc_energy_density(terms, ρ)
    Vρ = reshape(terms.vrho, s_ρ, n_p)
    Vρρ = terms.v2rho2

    δe = ntuple(Val(N)) do n
        libxc_assemble_δe(Vρ, get_partials_(ρ_δρ, n))
    end
    δVρ = ntuple(Val(N)) do n
        libxc_assemble_δVρ(Vρρ, get_partials_(ρ_δρ, n))
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
    e  = libxc_energy_density(terms, ρ)
    Vρ = reshape(terms.vrho,   s_ρ, n_p)
    Vσ = reshape(terms.vsigma, s_σ, n_p)
    Vρρ = terms.v2rho2
    Vρσ = terms.v2rhosigma
    Vσσ = terms.v2sigma2

    δe = ntuple(Val(N)) do n
        libxc_assemble_δe(Vρ, get_partials_(ρ_δρ, n),
                          Vσ, get_partials_(σ_δσ, n))
    end
    δVρ = ntuple(Val(N)) do n
        libxc_assemble_δVρ(Vρρ, get_partials_(ρ_δρ, n),
                           Vρσ, get_partials_(σ_δσ, n))
    end
    δVσ = ntuple(Val(N)) do n
        libxc_assemble_δVσ(Vρσ, get_partials_(ρ_δρ, n),
                           Vσσ, get_partials_(σ_δσ, n))
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
    e  = libxc_energy_density(terms, ρ)
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
        libxc_assemble_δe(Vρ, get_partials_(ρ_δρ, n),
                          Vσ, get_partials_(σ_δσ, n),
                          Vτ, get_partials_(τ_δτ, n))
    end
    δVρ = ntuple(Val(N)) do n
        libxc_assemble_δVρ(Vρρ, get_partials_(ρ_δρ, n),
                           Vρσ, get_partials_(σ_δσ, n),
                           Vρτ, get_partials_(τ_δτ, n))
    end
    δVσ = ntuple(Val(N)) do n
        libxc_assemble_δVσ(Vρσ, get_partials_(ρ_δρ, n),
                           Vσσ, get_partials_(σ_δσ, n),
                           Vστ, get_partials_(τ_δτ, n))
    end
    δVτ = ntuple(Val(N)) do n
        libxc_assemble_δVτ(Vρτ, get_partials_(ρ_δρ, n),
                           Vστ, get_partials_(σ_δσ, n),
                           Vττ, get_partials_(τ_δτ, n))
    end
    (; e=map(Dual{T},   e, δe...),
       Vρ=map(Dual{T}, Vρ, δVρ...),
       Vσ=map(Dual{T}, Vσ, δVσ...),
       Vτ=map(Dual{T}, Vτ, δVτ...))
end
@views function DftFunctionals.potential_terms(func::LibxcFunctional{:mggal},
                                               ρ_δρ::AbstractMatrix{DT},
                                               σ_δσ::AbstractMatrix{DT},
                                               τ_δτ::Union{Nothing,AbstractMatrix{DT}},
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
    e  = libxc_energy_density(terms, ρ)

    Vρ = reshape(terms.vrho,   s_ρ, n_p)
    Vσ = reshape(terms.vsigma, s_σ, n_p)
    Vτ = haskey(terms, :vtau) ? reshape(terms.vtau, s_ρ, n_p) : nothing
    Vl = reshape(terms.vlapl,  s_ρ, n_p)
    Vρρ = terms.v2rho2
    Vρσ = terms.v2rhosigma
    Vρτ = haskey(terms, :v2rhotau) ? terms.v2rhotau : nothing
    Vρl = terms.v2rholapl
    Vσσ = terms.v2sigma2
    Vστ = haskey(terms, :v2sigmatau) ? terms.v2sigmatau : nothing
    Vσl = terms.v2sigmalapl
    Vll = terms.v2lapl2
    Vlτ = haskey(terms, :v2lapltau) ? terms.v2lapltau : nothing
    Vττ = haskey(terms, :v2tau2) ? terms.v2tau2 : nothing

    δe = ntuple(Val(N)) do n
        libxc_assemble_δe(Vρ, get_partials_(ρ_δρ, n),
                          Vσ, get_partials_(σ_δσ, n),
                          Vτ, get_partials_(τ_δτ, n),
                          Vl, get_partials_(l_δl, n))
    end
    δVρ = ntuple(Val(N)) do n
        libxc_assemble_δVρ(Vρρ, get_partials_(ρ_δρ, n),
                           Vρσ, get_partials_(σ_δσ, n),
                           Vρτ, get_partials_(τ_δτ, n),
                           Vρl, get_partials_(l_δl, n))
    end
    δVσ = ntuple(Val(N)) do n
        libxc_assemble_δVσ(Vρσ, get_partials_(ρ_δρ, n),
                           Vσσ, get_partials_(σ_δσ, n),
                           Vστ, get_partials_(τ_δτ, n),
                           Vσl, get_partials_(l_δl, n))
    end
    δVl = ntuple(Val(N)) do n
        libxc_assemble_δVl(Vρl, get_partials_(ρ_δρ, n),
                           Vσl, get_partials_(σ_δσ, n),
                           Vlτ, get_partials_(τ_δτ, n),
                           Vll, get_partials_(l_δl, n))
    end

    if haskey(terms, :vtau)
        δVτ = ntuple(Val(N)) do n
            libxc_assemble_δVτ(Vρτ, get_partials_(ρ_δρ, n),
                               Vστ, get_partials_(σ_δσ, n),
                               Vττ, get_partials_(τ_δτ, n),
                               Vlτ, get_partials_(l_δl, n))
        end
        (; e=map(Dual{T},   e, δe...),
           Vρ=map(Dual{T}, Vρ, δVρ...),
           Vσ=map(Dual{T}, Vσ, δVσ...),
           Vτ=map(Dual{T}, Vτ, δVτ...),
           Vl=map(Dual{T}, Vl, δVl...))
    else
        (; e=map(Dual{T},   e, δe...),
           Vρ=map(Dual{T}, Vρ, δVρ...),
           Vσ=map(Dual{T}, Vσ, δVσ...),
           Vl=map(Dual{T}, Vl, δVl...))
    end
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
DftFunctionals.needs_τ(fun::DispatchFunctional)    = needs_τ(fun.inner)
DftFunctionals.needs_Δρ(fun::DispatchFunctional)   = needs_Δρ(fun.inner)

# Note: CuMatrix dispatch to Libxc.jl is defined in src/workarounds/cuda_arrays.jl
# Matrix element types that can dispatch to Libxc for potential computations
const LibxcDispatchFloat = Union{Float64,Dual{<:Any,Float64}}
function DftFunctionals.potential_terms(fun::DispatchFunctional, ρ::Matrix{<:LibxcDispatchFloat}, args...)
    potential_terms(fun.inner, ρ, args...)
end
function DftFunctionals.potential_terms(fun::DispatchFunctional, ρ::AbstractMatrix, args...)
    potential_terms(DftFunctional(identifier(fun)), ρ, args...)
end

# Matrix element types that can dispatch to Libxc for energy computations
const LibxcDispatchFloatEnergy = Union{LibxcDispatchFloat,Dual{<:Any,<:Dual{<:Any,Float64}}}
function DftFunctionals.energy_density(fun::DispatchFunctional, ρ::Matrix{<:LibxcDispatchFloatEnergy}, args...)
    energy_density(fun.inner, ρ, args...)
end
function DftFunctionals.energy_density(fun::DispatchFunctional, ρ::AbstractMatrix, args...)
    energy_density(DftFunctional(identifier(fun)), ρ, args...)
end

hybrid_parameters(::Functional{:lda})   = nothing
hybrid_parameters(::Functional{:gga})   = nothing
hybrid_parameters(::Functional{:mgga})  = nothing
hybrid_parameters(::Functional{:mggal}) = nothing
hybrid_parameters(fun::DispatchFunctional) = hybrid_parameters(fun.inner)
function hybrid_parameters(libxcfun::LibxcFunctional)
    fxc = Libxc.Functional(libxcfun.identifier)
    if Libxc.is_global_hybrid(fxc)
        exx_lr = exx_sr = fxc.exx_coefficient
        return (; exx_lr, exx_sr,
                  range_separation_parameter=nothing, range_separation_kernel=nothing)
    elseif Libxc.is_range_separated(fxc)
        exx_lr = fxc.cam_alpha
        exx_sr = fxc.cam_alpha + fxc.cam_beta

        if :hyb_lcy in fxc.flags || :hyb_camy in fxc.flags
            range_separation_kernel = :yukawa
        elseif :hyb_lc in fxc.flags || :hyb_cam in fxc.flags
            range_separation_kernel = :erf
        else
            error("Unknown range separation kernel")
        end
        return (; exx_lr, exx_sr, range_separation_kernel,
                  range_separation_parameter=fxc.cam_omega)
    else
        return nothing
    end
end
