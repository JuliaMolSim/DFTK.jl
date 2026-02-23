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

function libxc_unfold_spin(data::AbstractMatrix, n_spin::Int)
    n_p = size(data, 2)
    if n_spin == 1
        data  # Only one spin component
    elseif n_spin == 2
        unfolded = similar(data, 2, 2, n_p)
        unfolded[1, 1, :] = data[1, :]
        unfolded[2, 2, :] = data[3, :]

        unfolded[1, 2, :] = unfolded[2, 1, :] = data[2, :]
        unfolded
    elseif n_spin == 3
        unfolded = similar(data, 3, 3, n_p)
        unfolded[1, 1, :] = data[1, :]
        unfolded[2, 2, :] = data[4, :]
        unfolded[3, 3, :] = data[6, :]

        unfolded[1, 2, :] = unfolded[2, 1, :] = data[2, :]
        unfolded[1, 3, :] = unfolded[3, 1, :] = data[3, :]
        unfolded[2, 3, :] = unfolded[3, 2, :] = data[5, :]
        unfolded
    else
        error("Unspoorted n_spin == $n_spin")
    end
end

libxc_energy(terms, ρ) = haskey(terms, :zk) ? reshape(terms.zk, 1, size(ρ, 2)) .* sum(ρ, dims=1) : false

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
# - libxc returns the cross-spin derivatives in a compact form, see https://libxc.gitlab.io/manual/libxc-5.1.x/

# Combine N vectors of size (n_p) into one (N, n_p) array
libxc_combine_spins(xs...) = reduce(vcat, transpose.(xs))

@views function DftFunctionals.potential_terms(func::LibxcFunctional{:lda},
                                               ρ_δρ::AbstractMatrix{DT}) where {N,T,DT<:Dual{T,Float64,N}}
    ρ = ForwardDiff.value.(ρ_δρ)
    s_ρ, n_p = size(ρ)
    fun = Libxc.Functional(func.identifier; n_spin=s_ρ)
    derivatives = filter(in(Libxc.supported_derivatives(fun)), 0:2)
    terms = Libxc.evaluate(fun; rho=ρ, derivatives)
    e = libxc_energy(terms, ρ)
    Vρ = reshape(terms.vrho, s_ρ, n_p)

    δe = ntuple(Val(N)) do n
        sum(Vρ .* ForwardDiff.partials.(ρ_δρ, n); dims=1)
    end
    δVρ = ntuple(Val(N)) do n
        δρ = ForwardDiff.partials.(ρ_δρ, n)
        if s_ρ == 1
            terms.v2rho2 .* δρ
        else
            libxc_combine_spins(
                terms.v2rho2[1,:] .* δρ[1,:] .+ terms.v2rho2[2,:] .* δρ[2,:],
                terms.v2rho2[2,:] .* δρ[1,:] .+ terms.v2rho2[3,:] .* δρ[2,:],
            )
        end
    end
    (; e=map(Dual{T}, e, δe...),
       Vρ=map(Dual{T}, Vρ, δVρ...))
end
@views function DftFunctionals.potential_terms(func::LibxcFunctional{:gga},
                                               ρ_δρ::AbstractMatrix{DT},
                                               σ_δσ::AbstractMatrix{DT}) where {N,T,DT<:Dual{T,Float64,N}}
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

    δe = ntuple(Val(N)) do n
        ( sum(Vρ .* ForwardDiff.partials.(ρ_δρ, n); dims=1)
        + sum(Vσ .* ForwardDiff.partials.(σ_δσ, n); dims=1))
    end
    δVρ = ntuple(Val(N)) do n
        δρ = ForwardDiff.partials.(ρ_δρ, n)
        δσ = ForwardDiff.partials.(σ_δσ, n)
        if s_ρ == 1
            terms.v2rho2 .* δρ .+ terms.v2rhosigma .* δσ
        else
            libxc_combine_spins(
                # For all 2 ρ components: one line for ∂²/∂ρ², one line for ∂²/∂ρ∂σ
                terms.v2rho2[1,:] .* δρ[1,:] .+ terms.v2rho2[2,:] .* δρ[2,:]
                .+ terms.v2rhosigma[1,:] .* δσ[1,:] .+ terms.v2rhosigma[2,:] .* δσ[2,:] .+ terms.v2rhosigma[3,:] .* δσ[3,:],
                terms.v2rho2[2,:] .* δρ[1,:] .+ terms.v2rho2[3,:] .* δρ[2,:]
                .+ terms.v2rhosigma[4,:] .* δσ[1,:] .+ terms.v2rhosigma[5,:] .* δσ[2,:] .+ terms.v2rhosigma[6,:] .* δσ[3,:],
            )
        end
    end
    δVσ = ntuple(Val(N)) do n
        δρ = ForwardDiff.partials.(ρ_δρ, n)
        δσ = ForwardDiff.partials.(σ_δσ, n)
        if s_σ == 1
            terms.v2rhosigma .* δρ .+ terms.v2sigma2 .* δσ
        else
            libxc_combine_spins(
                # For all 3 σ components: one line for ∂²/∂σ∂ρ, one line for ∂²/∂σ²
                terms.v2rhosigma[1,:] .* δρ[1,:] .+ terms.v2rhosigma[4,:] .* δρ[2,:]
                .+ terms.v2sigma2[1,:] .* δσ[1,:] .+ terms.v2sigma2[2,:] .* δσ[2,:] .+ terms.v2sigma2[3,:] .* δσ[3,:],
                terms.v2rhosigma[2,:] .* δρ[1,:] .+ terms.v2rhosigma[5,:] .* δρ[2,:]
                .+ terms.v2sigma2[2,:] .* δσ[1,:] .+ terms.v2sigma2[4,:] .* δσ[2,:] .+ terms.v2sigma2[5,:] .* δσ[3,:],
                terms.v2rhosigma[3,:] .* δρ[1,:] .+ terms.v2rhosigma[6,:] .* δρ[2,:]
                .+ terms.v2sigma2[3,:] .* δσ[1,:] .+ terms.v2sigma2[5,:] .* δσ[2,:] .+ terms.v2sigma2[6,:] .* δσ[3,:],
            )
        end
    end

    (; e=map(Dual{T}, e, δe...),
       Vρ=map(Dual{T}, Vρ, δVρ...),
       Vσ=map(Dual{T}, Vσ, δVσ...))
end
# TODO mgga and mggal derivatives

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
function DftFunctionals.potential_terms(fun::DispatchFunctional,
                                        ρ::Matrix{<:Union{Float64,Dual{<:Any,Float64}}}, args...)
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
