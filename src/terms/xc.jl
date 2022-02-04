using Libxc
include("../xc/xc_evaluate.jl")

"""
Exchange-correlation term, defined by a list of functionals and usually evaluated through libxc.
"""
struct Xc
    functionals::Vector{Symbol}  # Symbols of the functionals (Libxc.jl / libxc convention)
    scaling_factor::Real         # Scales by an arbitrary factor (useful for exploration)

    # Density cutoff for XC computation: Below this value a gridpoint counts as zero
    # `nothing` implies that libxc defaults are used (for each functional a different
    # small positive value like 1e-24)
    density_threshold::Union{Nothing,Float64}

    # Threshold for potential terms: Below this value a potential term is counted as zero.
    potential_threshold::Float64
end
Xc(symbols::Symbol...; kwargs...) = Xc([symbols...]; kwargs...)
function Xc(symbols::Vector; scaling_factor=1, density_threshold=nothing, potential_threshold=0)
    Xc(convert.(Symbol, symbols), scaling_factor, density_threshold, potential_threshold)
end
function Base.show(io::IO, xc::Xc)
    fac = isone(xc.scaling_factor) ? "" : ", scaling_factor=$scaling_factor"
    fun = length(xc.functionals) == 1 ? ":$(xc.functionals[1])" : "$(xc.functionals)"
    print(io, "Xc($fun$fac)")
end

function (xc::Xc)(basis::PlaneWaveBasis{T}) where {T}
    isempty(xc.functionals) && return TermNoop()
    functionals = Functional.(xc.functionals; n_spin=basis.model.n_spin_components)
    if !isnothing(xc.density_threshold)
        for func in functionals
            func.density_threshold = xc.density_threshold
        end
    end
    TermXc(functionals, T(xc.scaling_factor), T(xc.potential_threshold))
end

struct TermXc <: TermNonlinear
    functionals::Vector{Functional}
    scaling_factor::Real
    potential_threshold::Real
end

@views @timing "ene_ops: xc" function ene_ops(term::TermXc, basis::PlaneWaveBasis{T},
                                              ψ, occ; ρ, τ=nothing, kwargs...) where {T}
    @assert !isempty(term.functionals)

    model  = basis.model
    n_spin = model.n_spin_components
    @assert all(xc.family in (:lda, :gga, :mgga) for xc in term.functionals)
    @assert all(xc.n_spin == n_spin for xc in term.functionals)

    # Compute kinetic energy density, if needed.
    if isnothing(τ) && any(is_mgga, term.functionals)
        if isnothing(ψ) || isnothing(occ)
            τ = zero(ρ)
        else
            τ = compute_kinetic_energy_density(basis, ψ, occ)
        end
    end

    # Take derivatives of the density, if needed.
    max_ρ_derivs = maximum(max_required_derivative, term.functionals)
    density = LibxcDensities(basis, max_ρ_derivs, ρ, τ)

    # Evaluate terms and energy contribution (zk == energy per unit particle)
    # It may happen that a functional does only provide a potenital and not an energy term
    # Therefore skip_unsupported_derivatives=true to avoid an error.
    terms = evaluate(term.functionals, density; skip_unsupported_derivatives=true)
    @assert haskey(terms, :vrho)
    if haskey(terms, :zk)
        E = sum(terms.zk .* ρ) * basis.dvol
    else
        E = zero(T)
    end

    # Map from the tuple of spin indices for the contracted density gradient
    # (s, t) to the index convention used in libxc (i.e. packed symmetry-adapted
    # storage), see details on "Spin-polarised calculations" below.
    tσ = libxc_spinindex_σ

    # Potential contributions Vρ -2 ∇⋅(Vσ ∇ρ) + ΔVl
    potential = zero(ρ)
    @views for s in 1:n_spin
        potential[:, :, :, s] .+= terms.vrho[s, :, :, :]
        if haskey(terms, :vsigma) && any(x -> abs(x) > term.potential_threshold, terms.vsigma)
            # Need gradient correction
            # TODO Drop do-block syntax here?
            potential[:, :, :, s] .+= -2divergence_real(basis) do α
                # Extra factor (1/2) for s != t is needed because libxc only keeps σ_{αβ}
                # in the energy expression. See comment block below on spin-polarised XC.
                sum((s == t ? one(T) : one(T)/2)
                    .* terms.vsigma[tσ(s, t), :, :, :] .* density.∇ρ_real[t, :, :, :, α]
                    for t in 1:n_spin)
            end
        end
        if haskey(terms, :vlapl) && any(x -> abs(x) > term.potential_threshold, terms.vlapl)
            @warn "Meta-GGAs with a Vlapl term have not yet been thoroughly tested." maxlog=1
            mG² = [-sum(abs2, G) for G in G_vectors_cart(basis)]
            Vl_fourier = r_to_G(basis, terms.vlapl[s, :, :, :])
            potential[:, :, :, s] .+= G_to_r(basis, mG² .* Vl_fourier)  # ΔVl
        end
    end

    # DivAgrad contributions -½ Vτ
    Vτ = nothing
    if haskey(terms, :vtau) && any(x -> abs(x) > term.potential_threshold, terms.vtau)
        # Need meta-GGA non-local operator (Note: -½ part of the definition of DivAgrid)
        Vτ = term.scaling_factor * permutedims(terms.vtau, (2, 3, 4, 1))
    end

    if term.scaling_factor != 1
        E *= term.scaling_factor
        potential .*= term.scaling_factor
        !isnothing(Vτ) && (Vτ .*= term.scaling_factor)
    end
    ops = map(basis.kpoints) do kpt
        if !isnothing(Vτ)
            [RealSpaceMultiplication(basis, kpt, potential[:, :, :, kpt.spin]),
             DivAgradOperator(basis, kpt, Vτ[:, :, :, kpt.spin])]
        else
            RealSpaceMultiplication(basis, kpt, potential[:, :, :, kpt.spin])
        end
    end
    (; E, ops)
end

#=  meta-GGA energy and potential

The total energy is
    Etot = ∫ ρ ε(ρ,σ,τ,Δρ)
where ε(ρ,σ,τ,Δρ) is the energy per unit particle, σ = |∇ρ|², τ = ½ ∑ᵢ |∇ϕᵢ|²
is the kinetic energy density and Δρ is the Laplacian of the density.

Libxc provides the scalars
    Vρ = ∂(ρ ε)/∂ρ
    Vσ = ∂(ρ ε)/∂σ
    Vτ = ∂(ρ ε)/∂τ
    Vl = ∂(ρ ε)/∂Δρ

Consider a variation δϕᵢ of an orbital (considered real for
simplicity), and let δEtot be the corresponding variation of the
energy. Then the potential Vxc is defined by
    δEtot = ∫ Vxc δρ = 2 ∫ Vxc ϕᵢ δϕᵢ

    δρ  = 2 ϕᵢ δϕᵢ
    δσ  = 2 ∇ρ  ⋅ ∇δρ = 4 ∇ρ ⋅ ∇(ϕᵢ δϕᵢ)
    δτ  =   ∇ϕᵢ ⋅ ∇δϕᵢ
    δΔρ = Δδρ = 2 Δ(ϕᵢ δϕᵢ)
    δEtot = ∫ Vρ δρ + Vσ δσ + Vτ δτ + Vl δΔρ
          = 2 ∫ Vρ ϕᵢ δϕᵢ + 4 ∫ Vσ ∇ρ ⋅ ∇(ϕᵢ δϕᵢ) +  ∫ Vτ ∇ϕᵢ ⋅ ∇δϕᵢ   + 2 ∫   Vl Δ(ϕᵢ δϕᵢ)
          = 2 ∫ Vρ ϕᵢ δϕᵢ - 4 ∫ div(Vσ ∇ρ) ϕᵢ δϕᵢ -  ∫ div(Vτ ∇ϕᵢ) δϕᵢ + 2 ∫ Δ(Vl)  ϕᵢ δϕᵢ
where we performed an integration by parts in the last tho equations
(boundary terms drop by periodicity). For GGA functionals we identify
    Vxc = Vρ - 2 div(Vσ ∇ρ),
see also Richard Martin, Electronic stucture, p. 158. For meta-GGAs an extra term ΔVl appears
and the Vτ term cannot be cast into a local potential form. We therefore define the
potential-orbital product as:
    Vxc ψ = [Vρ - 2 div(Vσ ∇ρ) + Δ(Vl)] ψ + div(-½Vτ ∇ψ)
=#

#=  Spin-polarised GGA calculations

These expressions can be generalised for spin-polarised calculations.
For simplicity we take GGA as an example, meta-GGA follows similarly.
In this case for example the energy per unit particle becomes
ε(ρ_α, ρ_β, σ_αα, σ_αβ, σ_βα, σ_ββ), where σ_ij = ∇ρ_i ⋅ ∇ρ_j
and the XC potential is analogously
    Vxc_s = Vρ_s - 2 ∑_t div(Vσ_{st} ∇ρ_t)
where s, t ∈ {α, β} are the spin components and we understand
    Vρ_s     = ∂(ρ ε)/∂(ρ_s)
    Vσ_{s,t} = ∂(ρ ε)/∂(σ_{s,t})

Now, in contrast to this libxc explicitly uses the symmetry σ_αβ = σ_βα and sets σ
to be a vector of the three independent components only
    σ = [σ_αα, σ_x, σ_ββ]  where     σ_x = (σ_αβ + σ_βα)/2
Accordingly Vσ has the components
    [∂(ρ ε)/∂σ_αα, ∂(ρ ε)/∂σ_x, ∂(ρ ε)/∂σ_ββ]
where in particular ∂(ρ ε)/∂σ_x = (1/2) ∂(ρ ε)/∂σ_αβ = (1/2) ∂(ρ ε)/∂σ_βα.
This explains the extra factor (1/2) needed in the GGA term of the XC potential
and which pops up in the GGA kernel whenever derivatives wrt. σ are considered.
=#

#=  Packed representation of spin-adapted libxc quantities.

When storing the spin components of the contracted density gradient as well
as the various derivatives of the energy wrt. ρ or σ, libxc uses a packed
representation exploiting spin symmetry. The following helper functions
allow to write more readable loops by taking care of the packing of a Cartesian
spin index to the libxc format.
=#

# Leaving aside the details with the identification of the second spin
# component of σ with (σ_αβ + σ_βα)/2 detailed above, the contracted density
# gradient σ seems to be storing the spin components [αα αβ ββ]. In DFTK we
# identify α with 1 and β with 2, leading to the spin mapping. The caller has
# to make sure to include a factor (1/2) in the contraction whenever s == t.
function libxc_spinindex_σ(s, t)
    s == 1 && t == 1 && return 1
    s == 2 && t == 2 && return 3
    return 2
end

# For terms.v2rho2 the spins are arranged as [(α, α), (α, β), (β, β)]
function libxc_spinindex_ρρ(s, t)
    s == 1 && t == 1 && return 1
    s == 2 && t == 2 && return 3
    return 2
end

# For e.g. terms.v2rhosigma the spins are arranged in row-major order as
# [(α, αα) (α, αβ) (α, ββ) (β, αα) (β, αβ) (β, ββ)]
# where the second entry in the tuple refers to the spin component of
# the σ derivative.
libxc_spinindex_ρσ(s, t) = @inbounds LinearIndices((3, 2))[t, s]

# For e.g. terms.v2sigma2 the spins are arranged as
# [(αα, αα) (αα, αβ) (αα, ββ) (αβ, αβ) (αβ, ββ) (ββ, ββ)]
function libxc_spinindex_σσ(s, t)
    s ≤ t || return libxc_spinindex_σσ(t, s)
    Dict((1, 1) => 1, (1, 2) => 2, (1, 3) => 3,
                      (2, 2) => 4, (2, 3) => 5,
                                   (3, 3) => 6
    )[(s, t)]
end

# TODO Hide some of the index and spin-factor details by wrapping around the terms tuple
#      returned from Libxc.evaluate ?

function max_required_derivative(functional)
    functional.family == :lda && return 0
    functional.family == :gga && return 1
    if functional.family == :mgga
        return needs_laplacian(functional) ? 2 : 1
    end
    error("Functional family $(functional.family) not known.")
end


# stores the input to libxc in a format it likes
struct LibxcDensities
    basis::PlaneWaveBasis
    max_derivative::Int
    ρ_real    # density ρ[iσ, ix, iy, iz]
    ∇ρ_real   # for GGA, density gradient ∇ρ[iσ, ix, iy, iz, iα]
    σ_real    # for GGA, contracted density gradient σ[iσ, ix, iy, iz]
    Δρ_real   # for (some) mGGA, Laplacian of the density Δρ[iσ, ix, iy, iz]
    τ_real    # Kinetic-energy density τ[iσ, ix, iy, iz]
end

"""
Compute density in real space and its derivatives starting from ρ
"""
function LibxcDensities(basis, max_derivative::Integer, ρ, τ)
    model = basis.model
    @assert max_derivative in (0, 1, 2)

    n_spin    = model.n_spin_components
    σ_real    = nothing
    ∇ρ_real   = nothing
    Δρ_real   = nothing

    # compute ρ_real and possibly ρ_fourier
    ρ_real = permutedims(ρ, (4, 1, 2, 3))  # ρ[x, y, z, σ] -> ρ_real[σ, x, y, z]
    if max_derivative > 0
        ρf = r_to_G(basis, ρ)
        ρ_fourier = permutedims(ρf, (4, 1, 2, 3))  # ρ_fourier[σ, x, y, z]
    end

    # compute ∇ρ and σ
    if max_derivative > 0
        n_spin_σ = div((n_spin + 1) * n_spin, 2)
        ∇ρ_real = similar(ρ_real,   n_spin, basis.fft_size..., 3)
        σ_real  = similar(ρ_real, n_spin_σ, basis.fft_size...)

        for α = 1:3
            iGα = [im * G[α] for G in G_vectors_cart(basis)]
            for σ = 1:n_spin
                ∇ρ_real[σ, :, :, :, α] .= G_to_r(basis, iGα .* @view ρ_fourier[σ, :, :, :])
            end
        end

        tσ = libxc_spinindex_σ  # Spin index transformation (s, t) => st as expected by Libxc
        σ_real .= 0
        @views for α in 1:3
            σ_real[tσ(1, 1), :, :, :] .+= ∇ρ_real[1, :, :, :, α] .* ∇ρ_real[1, :, :, :, α]
            if n_spin > 1
                σ_real[tσ(1, 2), :, :, :] .+= ∇ρ_real[1, :, :, :, α] .* ∇ρ_real[2, :, :, :, α]
                σ_real[tσ(2, 2), :, :, :] .+= ∇ρ_real[2, :, :, :, α] .* ∇ρ_real[2, :, :, :, α]
            end
        end
    end

    # Compute Δρ
    if max_derivative > 1
        Δρ_real = similar(ρ_real, n_spin, basis.fft_size...)
        mG² = [-sum(abs2, G) for G in G_vectors_cart(basis)]
        for σ = 1:n_spin
            Δρ_real[σ, :, :, :] .= G_to_r(basis, mG² .* @view ρ_fourier[σ, :, :, :])
        end
    end

    # τ[x, y, z, σ] -> τ_Libxc[σ, x, y, z]
    τ_Libxc = isnothing(τ) ? nothing : permutedims(τ, (4, 1, 2, 3))
    LibxcDensities(basis, max_derivative, ρ_real, ∇ρ_real, σ_real, Δρ_real, τ_Libxc)
end


function compute_kernel(term::TermXc, basis::PlaneWaveBasis; ρ, kwargs...)
    density = LibxcDensities(basis, 0, ρ, nothing)
    n_spin  = basis.model.n_spin_components
    @assert 1 ≤ n_spin ≤ 2
    if !all(xc.family == :lda for xc in term.functionals)
        error("compute_kernel only implemented for LDA")
    end
    @assert all(xc.n_spin == n_spin for xc in term.functionals)

    kernel = evaluate(term.functionals, density; derivatives=2:2).v2rho2
    fac = term.scaling_factor
    if n_spin == 1
        Diagonal(vec(fac .* kernel))
    else
        # Blocks in the kernel matrix mapping (ρα, ρβ) ↦ (Vα, Vβ)
        Kαα = @view kernel[1, :, :, :]
        Kαβ = @view kernel[2, :, :, :]
        Kβα = Kαβ
        Kββ = @view kernel[3, :, :, :]

        fac .* [Diagonal(vec(Kαα)) Diagonal(vec(Kαβ));
                Diagonal(vec(Kβα)) Diagonal(vec(Kββ))]
    end
end


function apply_kernel(term::TermXc, basis::PlaneWaveBasis{T}, δρ; ρ, kwargs...) where {T}
    n_spin = basis.model.n_spin_components
    isempty(term.functionals) && return nothing
    @assert all(xc.family in (:lda, :gga) for xc in term.functionals)
    @assert all(xc.n_spin == n_spin for xc in term.functionals)

    # Take derivatives of the density and the perturbation if needed.
    max_ρ_derivs = maximum(max_required_derivative, term.functionals)
    density      = LibxcDensities(basis, max_ρ_derivs, ρ, nothing)
    perturbation = LibxcDensities(basis, max_ρ_derivs, δρ, nothing)

    ∇ρ  = density.∇ρ_real
    δρ  = perturbation.ρ_real
    ∇δρ = perturbation.∇ρ_real

    # Compute required density / perturbation cross-derivatives
    cross_derivatives = Dict{Symbol, Any}()
    if max_ρ_derivs > 0
        cross_derivatives[:δσ] = [
            @views 2sum(∇ρ[I[1], :, :, :, α] .* ∇δρ[I[2], :, :, :, α] for α in 1:3)
            for I in CartesianIndices((n_spin, n_spin))
        ]
    end

    # TODO LDA actually only needs the 2nd derivatives for this ... could be optimised
    terms  = evaluate(term.functionals, density, derivatives=1:2)
    δV = zero(ρ)  # [iσ, ix, iy, iz]

    tρρ = libxc_spinindex_ρρ
    @views for s in 1:n_spin, t in 1:n_spin  # LDA term
        δV[:, :, :, s] .+= terms.v2rho2[tρρ(s, t), :, :, :] .* δρ[t, :, :, :]
    end
    if haskey(terms, :v2rhosigma)  # GGA term
        add_kernel_gradient_correction!(δV, terms, density, perturbation, cross_derivatives)
    end

    term.scaling_factor * δV
end


function add_kernel_gradient_correction!(δV, terms, density, perturbation, cross_derivatives)
    # Follows DOI 10.1103/PhysRevLett.107.216402
    #
    # For GGA V = Vρ - 2 ∇⋅(Vσ ∇ρ) = (∂ε/∂ρ) - 2 ∇⋅((∂ε/∂σ) ∇ρ)
    #
    # δV(r) = f(r,r') δρ(r') = (∂V/∂ρ) δρ + (∂V/∂σ) δσ
    #
    # therefore
    # δV(r) = (∂^2ε/∂ρ^2) δρ - 2 ∇⋅[(∂^2ε/∂σ∂ρ) ∇ρ + (∂ε/∂σ) (∂∇ρ/∂ρ)] δρ
    #       + (∂^2ε/∂ρ∂σ) δσ - 2 ∇⋅[(∂^ε/∂σ^2) ∇ρ  + (∂ε/∂σ) (∂∇ρ/∂σ)] δσ
    #
    # Note δσ = 2∇ρ⋅δ∇ρ = 2∇ρ⋅∇δρ, therefore
    #      - 2 ∇⋅((∂ε/∂σ) (∂∇ρ/∂σ)) δσ
    #    = - 2 ∇(∂ε/∂σ)⋅(∂∇ρ/∂σ) δσ - 2 (∂ε/∂σ) ∇⋅(∂∇ρ/∂σ) δσ
    #    = - 2 ∇(∂ε/∂σ)⋅δ∇ρ - 2 (∂ε/∂σ) ∇⋅δ∇ρ
    #    = - 2 ∇⋅((∂ε/∂σ) ∇δρ)
    # and (because assumed independent variables): (∂∇ρ/∂ρ) = 0.
    #
    # Note that below the LDA term (∂^2ε/∂ρ^2) δρ is not done here (dealt with by caller)

    basis  = density.basis
    n_spin = basis.model.n_spin_components
    ρ   = density.ρ_real
    ∇ρ  = density.∇ρ_real
    δρ  = perturbation.ρ_real
    ∇δρ = perturbation.∇ρ_real
    δσ  = cross_derivatives[:δσ]
    Vρσ = terms.v2rhosigma
    Vσσ = terms.v2sigma2
    Vσ  = terms.vsigma

    T   = eltype(ρ)
    tσ  = libxc_spinindex_σ
    tρσ = libxc_spinindex_ρσ
    tσσ = libxc_spinindex_σσ

    # Note: δV[iσ, ix, iy, iz] unlike the other quantities ...
    @views for s in 1:n_spin
        for t in 1:n_spin, u in 1:n_spin
            spinfac_tu = (t == u ? one(T) : one(T)/2)
            stu = tρσ(s, tσ(t, u))
            @. δV[:, :, :, s] += spinfac_tu * Vρσ[stu, :, :, :] * δσ[t, u][:, :, :]
        end

        # TODO Potential for some optimisation ... some contractions in this body are
        #      independent of α and could be precomputed.
        δV[:, :, :, s] .+= divergence_real(density.basis) do α
            ret_α = similar(density.ρ_real, basis.fft_size...)
            ret_α .= 0
            for t in 1:n_spin
                spinfac_st = (t == s ? one(T) : one(T)/2)
                ret_α .+= -2spinfac_st .* Vσ[tσ(s, t), :, :, :] .* ∇δρ[t, :, :, :, α]

                for u in 1:n_spin
                    spinfac_su = (s == u ? one(T) : one(T)/2)
                    tsu = tρσ(t, tσ(s, u))
                    ret_α .+= -2spinfac_su .* Vρσ[tsu, :, :, :] .* ∇ρ[u, :, :, :, α] .* δρ[t, :, :, :]

                    for v in 1:n_spin
                        spinfac_uv = (u == v ? one(T) : one(T)/2)
                        stuv = tσσ(tσ(s, t), tσ(u, v))
                        ret_α .+= (-2spinfac_uv .* spinfac_st .* Vσσ[stuv, :, :, :]
                                   .* ∇ρ[t, :, :, :, α] .* δσ[u, v][:, :, :])
                    end  # v
                end  # u
            end  # t
            ret_α
        end  # α
    end

    δV
end


function Libxc.evaluate(xc::Functional, density::LibxcDensities;
                        derivatives=0:1, skip_unsupported_derivatives=false, kwargs...)
    if skip_unsupported_derivatives
        derivatives = filter(i -> i in supported_derivatives(xc), derivatives)
    end
    if xc.family == :lda
        evaluate(xc; rho=density.ρ_real, derivatives, kwargs...)
    elseif xc.family == :gga
        evaluate(xc; rho=density.ρ_real, sigma=density.σ_real, derivatives, kwargs...)
    elseif xc.family == :mgga && !needs_laplacian(xc)
        evaluate(xc; rho=density.ρ_real, sigma=density.σ_real, tau=density.τ_real,
                 derivatives, kwargs...)
    elseif xc.family == :mgga && needs_laplacian(xc)
        evaluate(xc; rho=density.ρ_real, sigma=density.σ_real, tau=density.τ_real,
                 lapl=density.Δρ_real, derivatives, kwargs...)
    else
        error("Not implemented for functional familiy $(xc.family)")
    end
end
function Libxc.evaluate(xcs::Vector{Functional}, density::LibxcDensities; kwargs...)
    isempty(xcs) && return NamedTuple()
    result = evaluate(xcs[1], density; kwargs...)
    for i in 2:length(xcs)
        other = evaluate(xcs[i], density; kwargs...)
        for (key, data) in pairs(other)
            if haskey(result, key)
                result[key] .+= data
            else
                result = merge(result, (key => data, ))
            end
        end
    end
    result
end


"""
Compute divergence of an operand function, which returns the cartesian x,y,z
components in real space when called with the arguments 1 to 3.
The divergence is also returned as a real-space array.
"""
function divergence_real(operand, basis)
    gradsum = sum(1:3) do α
        operand_α = r_to_G(basis, operand(α))
        del_α = im * [G[α] for G in G_vectors_cart(basis)]
        del_α .* operand_α
    end
    G_to_r(basis, gradsum)
end
