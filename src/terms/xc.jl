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
end
Xc(symbols::Symbol...; kwargs...) = Xc([symbols...]; kwargs...)
function Xc(symbols::Vector; scaling_factor=1, density_threshold=nothing)
    Xc(convert.(Symbol, symbols), scaling_factor, density_threshold)
end

function (xc::Xc)(basis)
    functionals = Functional.(xc.functionals; n_spin=basis.model.n_spin_components)
    if !isnothing(xc.density_threshold)
        for func in functionals
            func.density_threshold = xc.density_threshold
        end
    end
    TermXc(basis, functionals, xc.scaling_factor)
end

struct TermXc <: Term
    basis::PlaneWaveBasis
    functionals::Vector{Functional}
    scaling_factor::Real
end

@timing "ene_ops: xc" function ene_ops(term::TermXc, ψ, occ; ρ, ρspin=nothing, kwargs...)
    basis = term.basis
    T     = eltype(basis)
    model = basis.model
    n_spin = model.n_spin_components
    @assert all(xc.family in (:lda, :gga) for xc in term.functionals)

    if isempty(term.functionals)
        ops = [NoopOperator(term.basis, kpoint) for kpoint in term.basis.kpoints]
        return (E=0, ops=ops)
    end

    # Take derivatives of the density if needed.
    max_ρ_derivs = maximum(max_required_derivative, term.functionals)
    density = LibxcDensity(basis, max_ρ_derivs, ρ, ρspin)

    potential = [zeros(T, basis.fft_size) for _ in 1:n_spin]  # TODO CPU arrays
    terms = evaluate(term.functionals, density)

    # Energy contribution (zk == energy per unit particle)
    dVol = basis.model.unit_cell_volume / prod(basis.fft_size)
    E = sum(terms.zk .* ρ.real) * dVol

    # Map from the cartesian index into the two spin indices of the contracted
    # density gradient σ_{αβ} to the convention used in libxc.
    tσ = libxc_spinindex_σ

    # Potential contributions Vρ -2 ∇⋅(Vσ ∇ρ)
    @views for s in 1:n_spin
        potential[s] .+= terms.vrho[s, :, :, :]

        if haskey(terms, :vsigma)  # Need gradient correction
            potential[s] .+= -2divergence_real(density.basis) do α
                # Extra factor (1/2) for s != t is needed because libxc only keeps σ_{αβ}
                # in the energy expression. See comment block below on spin-polarised XC.
                sum((s == t ? one(T) : one(T)/2)
                    .* terms.vsigma[tσ(s, t), :, :, :] .* density.∇ρ_real[t, :, :, :, α]
                    for t in 1:n_spin)
            end
        end
    end

    if term.scaling_factor != 1
        E *= term.scaling_factor
        potential = [pot .*= term.scaling_factor for pot in potential]
    end
    ops = [RealSpaceMultiplication(basis, kpoint, potential[kpoint.spin])
           for kpoint in basis.kpoints]
    (E=E, ops=ops)
end

# Kernel functions given towards the end of the file

#=
The total energy is
Etot = ∫ ρ E(ρ,σ), where σ = |∇ρ|^2 and E(ρ,σ) is the energy per unit particle.
libxc provides the scalars
    Vρ = ∂(ρ E)/∂ρ
    Vσ = ∂(ρ E)/∂σ

Consider a variation dϕi of an orbital (considered real for
simplicity), and let dEtot be the corresponding variation of the
energy. Then the potential Vxc is defined by
    dEtot = 2 ∫ Vxc ϕi dϕi

    dρ = 2 ϕi dϕi
    dσ = 2 ∇ρ ⋅ ∇dρ = 4 ∇ρ ⋅ ∇(ϕi dϕi)
    dEtot = ∫ Vρ dρ + Vσ dσ
          = 2 ∫ Vρ ϕi dϕi + 4 ∫ Vσ ∇ρ ⋅ ∇(ϕi dϕi)
          = 2 ∫ Vρ ϕi dϕi - 4 ∫ div(Vσ ∇ρ) ϕi dϕi
where we performed an integration by parts in the last equation
(boundary terms drop by periodicity). Therefore,
    Vxc = Vρ - 2 div(Vσ ∇ρ)
See eg Richard Martin, Electronic stucture, p. 158

A more algebraic way of deriving these for GGA, which helps see why
this is correct at the discrete level, is the following. Let

B the (Fourier, spherical) space of orbitals
C the (Fourier, cube) space of Fourier densities/potentials
C* the (real-space, cube) space of real-space densities/potentials

X : B -> C be the extension operator
R : C -> B the restriction (adjoint of X)
IF : C -> C* be the IFFT
F : C* -> be the FFT, adjoint of IF
K : C -> C be the gradient operator (multiplication by iK, assuming dimension 1 to
simplify notations)

Then
    ρ = IF(Xϕ).^2
    ρf = F ρ
    ∇ρf = K ρf
    ∇ρ = IF ∇ρf

    dρ  = 2 IF(Xϕ) .* IF(X dϕ)
    d∇ρ = IF K F dρ
        = 2 IF K F (IF(Xϕ) .* IF(X dϕ))

    Etot  = sum(ρ . * E(ρ, ∇ρ.^2))
    dEtot = sum(Vρ .* dρ + 2 (Vσ .* ∇ρ) .* d∇ρ)
          = sum(Vρ .* dρ + 2 (Vσ .* ∇ρ) .* [(IF K F) dρ)]
          = sum(Vρ .* dρ - 2 [(IF K F) (Vσ .* ∇ρ)] .* dρ)
where we used that (IF K F) is anti-self-adjoint and thought of sum(A .* op B )
like ⟨A | op B⟩; this is the discrete integration by parts. Now note that

    sum(V .* dρ) = 2 sum(V .* IF(Xϕ) .* IF(X dϕ))
                 = 2 sum(R(F[ V .* IF(Xϕ) ]) .* dϕ)

where we took the adjoint (IF X)^† = (R F) and the result follows.
=#

#=  Spin-polarised calculations

These expressions can be generalised for spin-polarised calculations.
In this case for example the energy per unit particle becomes
E(ρ_α, ρ_β, σ_αα, σ_αβ, σ_βα, σ_ββ), where σ_ij = ∇ρ_i ⋅ ∇ρ_j
and the XC potential is analogously
    Vxc_s = Vρ_s - 2 ∑_t div(Vσ_{st} ∇ρ_t)
where s, t ∈ {α, β} are the spin components and we understand
    Vρ_s     = ∂(ρ E)/∂(ρ_s)
    Vσ_{s,t} = ∂(ρ E)/∂(σ_{s,t})

Now, in contrast to this libxc explicitly uses the symmetry σ_αβ = σ_βα and sets σ
to be a vector of the three independent components only
    σ = [σ_αα, σ_x, σ_ββ]  where     σ_x = (σ_αβ + σ_βα)/2
Accordingly Vσ has the compoments
    [∂(ρ E)/∂σ_αα, ∂(ρ E)/∂σ_x, ∂(ρ E)/∂σ_ββ]
where in particular ∂(ρ E)/∂σ_x = (1/2) ∂(ρ E)/∂σ_αβ = (1/2) ∂(ρ E)/∂σ_βα.
This explains the extra factor (1/2) needed in the GGA term of the XC potential.
=#

function max_required_derivative(functional)
    functional.family == :lda && return 0
    functional.family == :gga && return 1
    error("Functional family $(functional.family) not known.")
end


struct LibxcDensity
    basis
    max_derivative::Int
    ρ_real    # density on real-space grid
    ∇ρ_real   # density gradient on real-space grid
    σ_real    # contracted density gradient on real-space grid
end

"""
Compute density in real space and its derivatives starting from ρ
"""
function LibxcDensity(basis, max_derivative::Integer, ρ::RealFourierArray, ρspin=nothing)
    model = basis.model
    @assert model.spin_polarization in (:collinear, :none, :spinless)
    @assert max_derivative in (0, 1)
    model.spin_polarization == :collinear && (@assert !isnothing(ρspin))
    ifft(x) = real_checked(G_to_r(basis, clear_without_conjugate!(x)))

    n_spin    = model.n_spin_components
    σ_real    = nothing
    ∇ρ_real   = nothing
    if model.spin_polarization == :collinear
        @assert n_spin == 2
        ρ_real    = similar(ρ.real,    n_spin, basis.fft_size...)
        ρ_real[1, :, :, :] .= @. (ρ.real + ρspin.real) / 2
        ρ_real[2, :, :, :] .= @. (ρ.real - ρspin.real) / 2

        if max_derivative > 0
            ρ_fourier = similar(ρ.fourier, n_spin, basis.fft_size...)
            ρ_fourier[1, :, :, :] .= @. (ρ.fourier + ρspin.fourier) / 2
            ρ_fourier[2, :, :, :] .= @. (ρ.fourier - ρspin.fourier) / 2
        end
    else
        @assert n_spin == 1
        ρ_real    = reshape(ρ.real,    1, basis.fft_size...)
        ρ_fourier = reshape(ρ.fourier, 1, basis.fft_size...)
    end

    if max_derivative > 0
        n_spin_σ = div((n_spin + 1) * n_spin, 2)
        ∇ρ_real = similar(ρ.real,   n_spin, basis.fft_size..., 3)
        σ_real  = similar(ρ.real, n_spin_σ, basis.fft_size...)

        for α = 1:3
            iGα = [im * G[α] for G in G_vectors_cart(basis)]
            for σ = 1:n_spin
                ∇ρ_real[σ, :, :, :, α] .= ifft(iGα .* @view ρ_fourier[σ, :, :, :])
            end
        end

        # Does the spin index transformation (σ, τ) => στ as expected by Libxc
        tσ = libxc_spinindex_σ

        σ_real .= 0
        @views for α in 1:3
            σ_real[tσ(1, 1), :, :, :] .+= ∇ρ_real[1, :, :, :, α] .* ∇ρ_real[1, :, :, :, α]
            if n_spin > 1
                σ_real[tσ(1, 2), :, :, :] .+= ∇ρ_real[1, :, :, :, α] .* ∇ρ_real[2, :, :, :, α]
                σ_real[tσ(2, 2), :, :, :] .+= ∇ρ_real[2, :, :, :, α] .* ∇ρ_real[2, :, :, :, α]
            end
        end
    end

    LibxcDensity(basis, max_derivative, ρ_real, ∇ρ_real, σ_real)
end

#
# XC kernel
#
function compute_kernel(term::TermXc; ρ::RealFourierArray, ρspin=nothing, kwargs...)
    @assert term.basis.model.spin_polarization in (:none, :spinless, :collinear)
    density = LibxcDensity(term.basis, 0, ρ, ρspin)
    n_spin = term.basis.model.n_spin_components
    @assert 1 ≤ n_spin ≤ 2
    if !all(xc.family == :lda for xc in term.functionals)
        error("compute_kernel only implemented for LDA")
    end

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

        # Blocks in the kernel matrix mapping (ρtot, ρspin) ↦ (Vα, Vβ)
        K_αtot  = Diagonal(vec(fac * (Kαα + Kαβ) / 2))
        K_αspin = Diagonal(vec(fac * (Kαα - Kαβ) / 2))
        K_βtot  = Diagonal(vec(fac * (Kβα + Kββ) / 2))
        K_βspin = Diagonal(vec(fac * (Kβα - Kββ) / 2))

        [K_αtot K_αspin;
         K_βtot K_βspin]
    end
end


function apply_kernel(term::TermXc, dρ::RealFourierArray, dρspin=nothing;
                      ρ::RealFourierArray, ρspin=nothing, kwargs...)
    basis  = term.basis
    T      = eltype(basis)
    n_spin = basis.model.n_spin_components
    isempty(term.functionals) && return nothing
    @assert all(xc.family in (:lda, :gga) for xc in term.functionals)
    @assert basis.model.spin_polarization in (:none, :spinless, :collinear)

    # Take derivatives of the density and the perturbation if needed.
    max_ρ_derivs = maximum(max_required_derivative, term.functionals)
    density      = LibxcDensity(basis, max_ρ_derivs, ρ, ρspin)
    perturbation = LibxcDensity(basis, max_ρ_derivs, dρ, dρspin)

    ∇ρ  = density.∇ρ_real
    dρ  = perturbation.ρ_real
    ∇dρ = perturbation.∇ρ_real

    # Compute required density / perturbation cross-derivatives
    cross_derivatives = Dict{Symbol, Any}()
    if max_ρ_derivs > 0
        cross_derivatives[:dσ] = [
            @views 2sum(∇ρ[I[1], :, :, :, α] .* ∇dρ[I[2], :, :, :, α] for α in 1:3)
            for I in CartesianIndices((n_spin, n_spin))
        ]
    end

    # TODO LDA actually only needs the 2nd derivatives for this ... could be optimised
    terms  = evaluate(term.functionals, density, derivatives=1:2)
    result = similar(ρ.real, basis.fft_size..., n_spin)
    result .= 0

    tρρ = libxc_spinindex_ρρ
    @views for s in 1:n_spin, t in 1:n_spin  # LDA term
        result[:, :, :, s] .+= terms.v2rho2[tρρ(s, t), :, :, :] .* dρ[t, :, :, :]
    end
    if haskey(terms, :v2rhosigma)  # GGA term
        add_kernel_gradient_correction!(result, terms, density, perturbation, cross_derivatives)
    end

    [from_real(basis, term.scaling_factor .* result[:, :, :, σ]) for σ in 1:n_spin]
end


function add_kernel_gradient_correction!(result, terms, density, perturbation, cross_derivatives)
    # Follows DOI 10.1103/PhysRevLett.107.216402
    #
    # For GGA V = Vρ - 2 ∇⋅(Vσ ∇ρ) = (∂ε/∂ρ) - 2 ∇⋅((∂ε/∂σ) ∇ρ)
    #
    # dV(r) = f(r,r') dρ(r') = (∂V/∂ρ) dρ + (∂V/∂σ) dσ
    #
    # therefore
    # dV(r) = (∂^2ε/∂ρ^2) dρ - 2 ∇⋅[(∂^2ε/∂σ∂ρ) ∇ρ + (∂ε/∂σ) (∂∇ρ/∂ρ)] dρ
    #       + (∂^2ε/∂ρ∂σ) dσ - 2 ∇⋅[(∂^ε/∂σ^2) ∇ρ  + (∂ε/∂σ) (∂∇ρ/∂σ)] dσ
    #
    # Note dσ = 2∇ρ⋅d∇ρ = 2∇ρ⋅∇dρ, therefore
    #      - 2 ∇⋅((∂ε/∂σ) (∂∇ρ/∂σ)) dσ
    #    = - 2 ∇(∂ε/∂σ)⋅(∂∇ρ/∂σ) dσ - 2 (∂ε/∂σ) ∇⋅(∂∇ρ/∂σ) dσ
    #    = - 2 ∇(∂ε/∂σ)⋅d∇ρ - 2 (∂ε/∂σ) ∇⋅d∇ρ
    #    = - 2 ∇⋅((∂ε/∂σ) ∇dρ)
    # and (because assumed independent variables): (∂∇ρ/∂ρ) = 0.
    #
    # Note that below the LDA term (∂^2ε/∂ρ^2) dρ is not done (dealt with by caller)

    basis  = density.basis
    n_spin = basis.model.n_spin_components
    ρ   = density.ρ_real
    ∇ρ  = density.∇ρ_real
    dρ  = perturbation.ρ_real
    ∇dρ = perturbation.∇ρ_real
    dσ  = cross_derivatives[:dσ]
    Vρσ = terms.v2rhosigma
    Vσσ = terms.v2sigma2
    Vσ  = terms.vsigma

    T   = eltype(ρ)
    tσ  = libxc_spinindex_σ
    tρσ = libxc_spinindex_ρσ
    tσσ = libxc_spinindex_σσ

    @views for s in 1:n_spin
        for t in 1:n_spin, u in 1:n_spin
            spinfac_tu = (t == u ? one(T) : one(T)/2)
            stu = tρσ(s, tσ(t, u))
            @. result[:, :, :, s] += spinfac_tu * Vρσ[stu, :, :, :] * dσ[t, u][:, :, :]
        end

        # TODO Potential for some optimisation ... some contractions in this body are
        #      independent of α and could be precomputed.
        result[:, :, :, s] .+= divergence_real(density.basis) do α
            ret_α = similar(density.ρ_real, basis.fft_size...)
            ret_α .= 0
            for t in 1:n_spin
                spinfac_st = (t == s ? one(T) : one(T)/2)
                ret_α .+= -2spinfac_st .* Vσ[tσ(s, t), :, :, :] .* ∇dρ[t, :, :, :, α]

                for u in 1:n_spin
                    spinfac_su = (s == u ? one(T) : one(T)/2)
                    tsu = tρσ(t, tσ(s, u))
                    ret_α .+= -2spinfac_su .* Vρσ[tsu, :, :, :] .* ∇ρ[u, :, :, :, α] .* dρ[t, :, :, :]

                    for v in 1:n_spin
                        spinfac_uv = (u == v ? one(T) : one(T)/2)
                        stuv = tσσ(tσ(s, t), tσ(u, v))
                        ret_α .+= (-2spinfac_uv .* spinfac_st .* Vσσ[stuv, :, :, :]
                                   .* ∇ρ[t, :, :, :, α] .* dσ[u, v][:, :, :])
                    end  # v
                end  # u
            end  # t
            ret_α
        end  # α
    end

    result
end


#
# Details
#
function Libxc.evaluate(xc::Functional, density::LibxcDensity; kwargs...)
    if xc.family == :lda
        evaluate(xc; rho=density.ρ_real, kwargs...)
    elseif xc.family == :gga
        evaluate(xc; rho=density.ρ_real, sigma=density.σ_real, kwargs...)
    else
        error("Not implemented for functional familiy $(xc.family)")
    end
end
function Libxc.evaluate(xcs::Vector{Functional}, density::LibxcDensity; kwargs...)
    isempty(xcs) && return NamedTuple()
    @assert all(xc.family == xcs[1].family for xc in xcs)

    result = evaluate(xcs[1], density; kwargs...)
    for i in 2:length(xcs)
        other = evaluate(xcs[i], density; kwargs...)
        for (k, v) in pairs(other)
            result[k] .+= other[k]
        end
    end
    result
end


# Libxc treats the contracted density gradient σ like an object with 3 spins.
# This translates between the "cartesian" representation of a tuple of two
# spin indices (s, t) to the representation used in libxc. Assumes s, t
# only take the values 1 and 2.
function libxc_spinindex_σ(s, t)
    s == 1 && t == 1 && return 1
    s == 2 && t == 2 && return 3
    return 2
end

# Symmetrised spin index for a quantity containing two ρ-like spinindices
# (like the second derivative of the energy wrt. ρ)
function libxc_spinindex_ρρ(s, t)
    s == 1 && t == 1 && return 1
    s == 2 && t == 2 && return 3
    return 2
end

libxc_spinindex_ρσ(s, t) = @inbounds LinearIndices((3, 2))[t, s]

function libxc_spinindex_σσ(s, t)
    s ≤ t || return libxc_spinindex_σσ(t, s)
    Dict((1, 1) => 1, (1, 2) => 2, (1, 3) => 3,
                      (2, 2) => 4, (2, 3) => 5,
                                   (3, 3) => 6
    )[(s, t)]
end


"""
Compute divergence of an operand function, which returns the cartesian x,y,z
components in real space when called with the arguments 1 to 3.
The divergence is also returned as a real-space array.
"""
function divergence_real(operand, basis)
    gradsum = sum(1:3) do α
        operand_α = r_to_G(basis, complex(operand(α)))
        del_α = im * [G[α] for G in G_vectors_cart(basis)]
        del_α .* operand_α
    end
    real(G_to_r(basis, gradsum))
end
