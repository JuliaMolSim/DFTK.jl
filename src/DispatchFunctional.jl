using DftFunctionals
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

    @assert fun.family in (:lda, :gga, :mgga)  # Hybrids not supported yet.
    if fun.family == :mgga && Libxc.needs_laplacian(fun)
        family = :mggal
    else
        family = fun.family
    end
    LibxcFunctional{family,kind}(identifier)
end

DftFunctionals.identifier(fun::LibxcFunctional) = fun.identifier
function DftFunctionals.has_energy(func::LibxcFunctional)
    0 in Libxc.supported_derivatives(Libxc.Functional(func.identifier))
end

function libxc_unfold_spin(data::Matrix, n_spin::Int)
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

libxc_energy(terms, ρ) = haskey(terms, :zk) ? reshape(terms.zk, 1, size(ρ, 2)) .* ρ : false

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

function DftFunctionals.kernel_terms(func::LibxcFunctional{:lda}, ρ::AbstractMatrix{Float64})
    s_ρ, n_p = size(ρ)
    fun = Libxc.Functional(func.identifier; n_spin=s_ρ)
    derivatives = filter(in(Libxc.supported_derivatives(fun)), 0:2)
    terms = Libxc.evaluate(fun; rho=ρ, derivatives)
    e   = libxc_energy(terms, ρ)
    Vρ  = reshape(terms.vrho,   s_ρ, n_p)
    Vρρ = libxc_unfold_spin(terms.v2rho2, s_ρ)
    (; e, Vρ, Vρρ)
end
function DftFunctionals.kernel_terms(func::LibxcFunctional{:gga}, ρ::AbstractMatrix{Float64},
                                     σ::AbstractMatrix{Float64})
    s_ρ, n_p = size(ρ)
    s_σ = size(σ, 1)
    fun = Libxc.Functional(func.identifier; n_spin=s_ρ)
    derivatives = filter(in(Libxc.supported_derivatives(fun)), 0:2)
    terms = Libxc.evaluate(fun; rho=ρ, sigma=σ, derivatives)
    e   = libxc_energy(terms, ρ)
    Vρ  = reshape(terms.vrho,   s_ρ, n_p)
    Vσ  = reshape(terms.vsigma, s_σ, n_p)
    Vρρ = libxc_unfold_spin(terms.v2rho2, s_ρ)
    Vρσ = permutedims(reshape(terms.v2rhosigma, s_σ, s_ρ, n_p), (2, 1, 3))
    Vσσ = libxc_unfold_spin(terms.v2sigma2, s_σ)
    (; e, Vρ, Vσ, Vρρ, Vρσ, Vσσ)
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

for fun in (:potential_terms, :kernel_terms)
    @eval begin
        # Note: CuMatrix dispatch to Libxc.jl is defined in src/workarounds/cuda_arrays.jl
        function DftFunctionals.$fun(fun::DispatchFunctional, ρ::Matrix{Float64}, args...)
            $fun(fun.inner, ρ, args...)
        end
        function DftFunctionals.$fun(fun::DispatchFunctional, ρ::AbstractMatrix, args...)
            $fun(DftFunctional(identifier(fun)), ρ, args...)
        end
    end
end
