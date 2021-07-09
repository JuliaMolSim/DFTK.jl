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

@views @timing "ene_ops: xc" function ene_ops(term::TermXc, ψ, occ; ρ, kwargs...)
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
    density = LibxcDensity(basis, max_ρ_derivs, ρ)

    potential = zero(ρ)
    terms = evaluate(term.functionals, density)

    # Energy contribution (zk == energy per unit particle)
    E = sum(terms.zk .* ρ) * term.basis.dvol

    # Map from the tuple of spin indices for the contracted density gradient
    # (s, t) to the index convention used in libxc (i.e. packed symmetry-adapted
    # storage), see details on "Spin-polarised calculations" below.
    tσ = libxc_spinindex_σ

    # Potential contributions Vρ -2 ∇⋅(Vσ ∇ρ)
    @views for s in 1:n_spin
        potential[:, :, :, s] .+= terms.vrho[s, :, :, :]

        if haskey(terms, :vsigma)  # Need gradient correction
            # TODO Drop do-block syntax here?
            potential[:, :, :, s] .+= -2divergence_real(density.basis) do α
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
        potential .*= term.scaling_factor
    end
    ops = [RealSpaceMultiplication(basis, kpoint, potential[:, :, :, kpoint.spin])
           for kpoint in basis.kpoints]
    (E=E, ops=ops)
end

#=  GGA energy and potential

The total energy is
Etot = ∫ ρ E(ρ,σ), where σ = |∇ρ|^2 and E(ρ,σ) is the energy per unit particle.
libxc provides the scalars
    Vρ = ∂(ρ E)/∂ρ
    Vσ = ∂(ρ E)/∂σ

Consider a variation δϕi of an orbital (considered real for
simplicity), and let δEtot be the corresponding variation of the
energy. Then the potential Vxc is defined by
    δEtot = 2 ∫ Vxc ϕi δϕi

    δρ = 2 ϕi δϕi
    δσ = 2 ∇ρ ⋅ ∇δρ = 4 ∇ρ ⋅ ∇(ϕi δϕi)
    δEtot = ∫ Vρ δρ + Vσ δσ
          = 2 ∫ Vρ ϕi δϕi + 4 ∫ Vσ ∇ρ ⋅ ∇(ϕi δϕi)
          = 2 ∫ Vρ ϕi δϕi - 4 ∫ div(Vσ ∇ρ) ϕi δϕi
where we performed an integration by parts in the last equation
(boundary terms drop by periodicity). Therefore,
    Vxc = Vρ - 2 div(Vσ ∇ρ)
See eg Richard Martin, Electronic stucture, p. 158
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
    error("Functional family $(functional.family) not known.")
end


# stores the input to libxc in a format it likes
struct LibxcDensity
    basis
    max_derivative::Int
    ρ_real    # density ρ[iσ, ix, iy, iz]
    ∇ρ_real   # for GGA, density gradient ∇ρ[iσ, ix, iy, iz, iα]
    σ_real    # for GGA, contracted density gradient σ[iσ, ix, iy, iz]
end

"""
Compute density in real space and its derivatives starting from ρ
"""
function LibxcDensity(basis, max_derivative::Integer, ρ)
    model = basis.model
    @assert max_derivative in (0, 1)

    n_spin    = model.n_spin_components
    σ_real    = nothing
    ∇ρ_real   = nothing

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

    LibxcDensity(basis, max_derivative, ρ_real, ∇ρ_real, σ_real)
end


function compute_kernel(term::TermXc; ρ, kwargs...)
    density = LibxcDensity(term.basis, 0, ρ)
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

        fac .* [Diagonal(vec(Kαα)) Diagonal(vec(Kαβ));
                Diagonal(vec(Kβα)) Diagonal(vec(Kββ))]
    end
end


function apply_kernel(term::TermXc, δρ; ρ, kwargs...)
    basis  = term.basis
    T      = eltype(basis)
    n_spin = basis.model.n_spin_components
    isempty(term.functionals) && return nothing
    @assert all(xc.family in (:lda, :gga) for xc in term.functionals)

    # Take derivatives of the density and the perturbation if needed.
    max_ρ_derivs = maximum(max_required_derivative, term.functionals)
    density      = LibxcDensity(basis, max_ρ_derivs, ρ)
    perturbation = LibxcDensity(basis, max_ρ_derivs, δρ)

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
