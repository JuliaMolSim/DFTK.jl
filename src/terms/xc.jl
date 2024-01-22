"""
Exchange-correlation term, defined by a list of functionals and usually evaluated through libxc.
"""
struct Xc
    functionals::Vector{Functional}
    scaling_factor::Real  # Scales by an arbitrary factor (useful for exploration)

    # Threshold for potential terms: Below this value a potential term is counted as zero.
    potential_threshold::Real

    # Use non-linear core correction or not
    use_nlcc::Bool
end
function Xc(functionals::AbstractVector{<:Functional}; scaling_factor=1,
            potential_threshold=0, use_nlcc=true)
    Xc(functionals, scaling_factor, potential_threshold, use_nlcc)
end
function Xc(functionals::AbstractVector; kwargs...)
    fun = map(functionals) do f
        f isa Functional ? f : DispatchFunctional(f)
    end
    Xc(convert(Vector{Functional}, fun); kwargs...)
end
Xc(functional; kwargs...) = Xc([functional]; kwargs...)

function Base.show(io::IO, xc::Xc)
    fac = isone(xc.scaling_factor) ? "" : ", scaling_factor=$(xc.scaling_factor)"
    fun = join(xc.functionals, ", ")
    print(io, "Xc($fun$fac)")
end

function (xc::Xc)(basis::PlaneWaveBasis{T}) where {T}
    isempty(xc.functionals) && return TermNoop()

    # Charge density for non-linear core correction
    ρcore = nothing
    if xc.use_nlcc && any(has_core_density, basis.model.atoms)
        ρcore = ρ_from_total(basis, atomic_total_density(basis, CoreDensity()))
        minimum(ρcore) < -sqrt(eps(T)) && @warn("Negative ρcore detected: $(minimum(ρcore))")
    end
    functionals = map(xc.functionals) do fun
        # Strip duals from functional parameters if needed
        newparams = convert_dual.(T, parameters(fun))
        change_parameters(fun, newparams; keep_identifier=true)
    end
    TermXc(convert(Vector{Functional}, functionals),
           convert_dual(T, xc.scaling_factor),
           T(xc.potential_threshold), ρcore)
end

struct TermXc{T,CT} <: TermNonlinear where {T,CT}
    functionals::Vector{Functional}
    scaling_factor::T
    potential_threshold::T
    ρcore::CT
end

function xc_potential_real(term::TermXc, basis::PlaneWaveBasis{T}, ψ, occupation;
                           ρ, τ=nothing) where {T}
    @assert !isempty(term.functionals)

    model    = basis.model
    n_spin   = model.n_spin_components
    potential_threshold = term.potential_threshold
    @assert all(family(xc) in (:lda, :gga, :mgga, :mggal) for xc in term.functionals)

    # Add the model core charge density (non-linear core correction)
    if !isnothing(term.ρcore)
        ρ = ρ + term.ρcore
    end

    # Compute kinetic energy density, if needed.
    if isnothing(τ) && any(needs_τ, term.functionals)
        if isnothing(ψ) || isnothing(occupation)
            τ = zero(ρ)
        else
            τ = compute_kinetic_energy_density(basis, ψ, occupation)
        end
    end

    # Take derivatives of the density, if needed.
    max_ρ_derivs = maximum(max_required_derivative, term.functionals)
    density = LibxcDensities(basis, max_ρ_derivs, ρ, τ)

    # Evaluate terms and energy contribution
    # If the XC functional is not supported for an architecture, terms is on the CPU
    terms = potential_terms(term.functionals, density)
    @assert haskey(terms, :Vρ) && haskey(terms, :e)
    E = term.scaling_factor * sum(terms.e) * basis.dvol

    # Map from the tuple of spin indices for the contracted density gradient
    # (s, t) to the index convention used in DftFunctionals (i.e. packed symmetry-adapted
    # storage), see details on "Spin-polarised calculations" below.
    tσ = DftFunctionals.spinindex_σ

    # Potential contributions Vρ -2 ∇⋅(Vσ ∇ρ) + ΔVl
    potential = zero(ρ)
    @views for s = 1:n_spin
        Vρ = to_device(basis.architecture, reshape(terms.Vρ, n_spin, basis.fft_size...))

        potential[:, :, :, s] .+= Vρ[s, :, :, :]
        if haskey(terms, :Vσ) && any(x -> abs(x) > potential_threshold, terms.Vσ)
            # Need gradient correction
            # TODO Drop do-block syntax here?
            potential[:, :, :, s] .+= -2divergence_real(basis) do α
                Vσ = to_device(basis.architecture, reshape(terms.Vσ, :, basis.fft_size...))

                # Extra factor (1/2) for s != t is needed because libxc only keeps σ_{αβ}
                # in the energy expression. See comment block below on spin-polarised XC.
                sum((s == t ? one(T) : one(T)/2)
                    .* Vσ[tσ(s, t), :, :, :] .* density.∇ρ_real[t, :, :, :, α]
                    for t = 1:n_spin)
            end
        end
        if haskey(terms, :Vl) && any(x -> abs(x) > potential_threshold, terms.Vl)
            @warn "Meta-GGAs with a Δρ term have not yet been thoroughly tested." maxlog=1
            mG² = .-norm2.(G_vectors_cart(basis))
            Vl  = to_device(basis.architecture, reshape(terms.Vl, n_spin, basis.fft_size...))
            Vl_fourier = fft(basis, Vl[s, :, :, :])
            potential[:, :, :, s] .+= irfft(basis, mG² .* Vl_fourier)  # ΔVl
        end
    end

    # DivAgrad contributions -½ Vτ
    Vτ = nothing
    if haskey(terms, :Vτ) && any(x -> abs(x) > potential_threshold, terms.Vτ)
        # Need meta-GGA non-local operator (Note: -½ part of the definition of DivAgrid)
        Vτ = to_device(basis.architecture, reshape(terms.Vτ, n_spin, basis.fft_size...))
        Vτ = term.scaling_factor * permutedims(Vτ, (2, 3, 4, 1))
    end

    # Note: We always have to do this, otherwise we get issues with AD wrt. scaling_factor
    potential .*= term.scaling_factor

    (; E, potential, Vτ)
end

@views @timing "ene_ops: xc" function ene_ops(term::TermXc, basis::PlaneWaveBasis{T},
                                              ψ, occupation; ρ, τ=nothing,
                                              kwargs...) where {T}
    E, Vxc, Vτ = xc_potential_real(term, basis, ψ, occupation; ρ, τ)

    ops = map(basis.kpoints) do kpt
        if !isnothing(Vτ)
            [RealSpaceMultiplication(basis, kpt, Vxc[:, :, :, kpt.spin]),
             DivAgradOperator(basis, kpt, Vτ[:, :, :, kpt.spin])]
        else
            RealSpaceMultiplication(basis, kpt, Vxc[:, :, :, kpt.spin])
        end
    end
    (; E, ops)
end

@timing "forces: xc" function compute_forces(term::TermXc, basis::PlaneWaveBasis{T},
                                             ψ, occupation; ρ, τ=nothing,
                                             kwargs...) where {T}
    # The only non-zero force contribution is from the nlcc core charge:
    # early return if nlcc is disabled / no elements have model core charges.
    isnothing(term.ρcore) && return nothing

    Vxc_real = xc_potential_real(term, basis, ψ, occupation; ρ, τ).potential
    # TODO: the factor of 2 here should be associated with the density, not the potential
    if basis.model.spin_polarization in (:none, :spinless)
        Vxc_fourier = fft(basis, Vxc_real[:,:,:,1])
    else
        Vxc_fourier = fft(basis, mean(Vxc_real, dims=4))
    end

    model = basis.model
    form_factors = atomic_density_form_factors(basis, CoreDensity())
    nlcc_groups = [(igroup, group) for (igroup, group) in enumerate(basis.model.atom_groups)
                   if has_core_density(model.atoms[first(group)])]
    @assert !isnothing(nlcc_groups)

    TT = promote_type(T, eltype(Vxc_real))
    forces = [zero(Vec3{TT}) for _ = 1:length(model.positions)]
    for (igroup, group) in nlcc_groups
        for iatom in group
            r = model.positions[iatom]
            forces[iatom] = _force_xc(basis, Vxc_fourier, form_factors, igroup, r)
        end
    end
    forces
end

# Function barrier to work around various type instabilities.
function _force_xc(basis::PlaneWaveBasis{T}, Vxc_fourier::AbstractArray{U}, form_factors,
                   igroup, r) where {T, U}
    TT = promote_type(T, real(U))
    f  = zero(Vec3{TT})
    for (iG, (G, G_cart)) in enumerate(zip(G_vectors(basis), G_vectors_cart(basis)))
        f -= real(conj(Vxc_fourier[iG])
                  .* form_factors[(igroup, norm(G_cart))]
                  .* cis2pi(-dot(G, r))
                  .* (-2T(π)) .* G .* im
                  ./ sqrt(basis.model.unit_cell_volume))
    end
    f
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

In particular this leads to an extra factor (1/2) which needs to be included
whenever using derivatives wrt. the off-diagonal component `σ_x` as a replacement
for derivatives wrt. σ_αβ or σ_βα.
=#

function max_required_derivative(functional)
    family(functional) == :lda   && return 0
    family(functional) == :gga   && return 1
    family(functional) == :mgga  && return 1
    family(functional) == :mggal && return 2
    error("Functional family $(family(functional)) not known.")
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
        ρf = fft(basis, ρ)
        ρ_fourier = permutedims(ρf, (4, 1, 2, 3))  # ρ_fourier[σ, x, y, z]
    end

    # compute ∇ρ and σ
    if max_derivative > 0
        n_spin_σ = div((n_spin + 1) * n_spin, 2)
        ∇ρ_real = similar(ρ_real,   n_spin, basis.fft_size..., 3)
        σ_real  = similar(ρ_real, n_spin_σ, basis.fft_size...)

        for α = 1:3
            iGα = map(G -> im * G[α], G_vectors_cart(basis))
            for σ = 1:n_spin
                ∇ρ_real[σ, :, :, :, α] .= irfft(basis, iGα .* @view ρ_fourier[σ, :, :, :])
            end
        end

        tσ = DftFunctionals.spinindex_σ  # Spin index transformation (s, t) => st as expected by Libxc
        σ_real .= 0
        @views for α = 1:3
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
        mG² = .-norm2.(G_vectors_cart(basis))
        for σ = 1:n_spin
            Δρ_real[σ, :, :, :] .= irfft(basis, mG² .* @view ρ_fourier[σ, :, :, :])
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
    if !all(family(xc) == :lda for xc in term.functionals)
        error("compute_kernel only implemented for LDA")
    end

    kernel = kernel_terms(term.functionals, density).Vρρ
    fac = term.scaling_factor
    if n_spin == 1
        Diagonal(vec(fac .* kernel))
    else
        # Blocks in the kernel matrix mapping (ρα, ρβ) ↦ (Vα, Vβ)
        Kαα = @view kernel[1, 1, :, :, :]
        Kαβ = @view kernel[1, 2, :, :, :]
        Kβα = @view kernel[2, 1, :, :, :]
        Kββ = @view kernel[2, 2, :, :, :]

        fac .* [Diagonal(vec(Kαα)) Diagonal(vec(Kαβ));
                Diagonal(vec(Kβα)) Diagonal(vec(Kββ))]
    end
end


function apply_kernel(term::TermXc, basis::PlaneWaveBasis{T}, δρ::AbstractArray{Tδρ};
                      ρ, kwargs...) where {T, Tδρ}
    n_spin = basis.model.n_spin_components
    isempty(term.functionals) && return nothing
    @assert all(family(xc) in (:lda, :gga) for xc in term.functionals)

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
            @views 2sum(∇ρ[I[1], :, :, :, α] .* ∇δρ[I[2], :, :, :, α] for α = 1:3)
            for I in CartesianIndices((n_spin, n_spin))
        ]
    end

    # If the XC functional is not supported for an architecture, terms is on the CPU
    terms = kernel_terms(term.functionals, density)
    δV = zeros(Tδρ, size(ρ)...)  # [ix, iy, iz, iσ]

    Vρρ = to_device(basis.architecture, reshape(terms.Vρρ, n_spin, n_spin, basis.fft_size...))
    @views for s = 1:n_spin, t = 1:n_spin  # LDA term
        δV[:, :, :, s] .+= Vρρ[s, t, :, :, :] .* δρ[t, :, :, :]
    end
    if haskey(terms, :Vρσ)  # GGA term
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
    spin_σ = 2n_spin - 1
    ρ   = density.ρ_real
    ∇ρ  = density.∇ρ_real
    δρ  = perturbation.ρ_real
    ∇δρ = perturbation.∇ρ_real
    δσ  = cross_derivatives[:δσ]
    Vρσ = to_device(basis.architecture, reshape(terms.Vρσ, n_spin, spin_σ, basis.fft_size...))
    Vσσ = to_device(basis.architecture, reshape(terms.Vσσ, spin_σ, spin_σ, basis.fft_size...))
    Vσ  = to_device(basis.architecture, reshape(terms.Vσ,  spin_σ,         basis.fft_size...))

    T   = eltype(ρ)
    tσ  = DftFunctionals.spinindex_σ

    # Note: δV[ix, iy, iz, iσ] unlike the other quantities ...
    @views for s = 1:n_spin
        for t = 1:n_spin, u = 1:n_spin
            spinfac_tu = (t == u ? one(T) : one(T)/2)
            @. δV[:, :, :, s] += spinfac_tu * Vρσ[s, tσ(t, u), :, :, :] * δσ[t, u][:, :, :]
        end

        # TODO Potential for some optimisation ... some contractions in this body are
        #      independent of α and could be precomputed.
        δV[:, :, :, s] .+= divergence_real(density.basis) do α
            ret_α = similar(density.ρ_real, basis.fft_size...)
            ret_α .= 0
            for t = 1:n_spin
                spinfac_st = (t == s ? one(T) : one(T)/2)
                ret_α .+= -2spinfac_st .* Vσ[tσ(s, t), :, :, :] .* ∇δρ[t, :, :, :, α]

                for u = 1:n_spin
                    spinfac_su = (s == u ? one(T) : one(T)/2)
                    ret_α .+= (-2spinfac_su .* Vρσ[t, tσ(s, u), :, :, :]
                               .* ∇ρ[u, :, :, :, α] .* δρ[t, :, :, :])

                    for v = 1:n_spin
                        spinfac_uv = (u == v ? one(T) : one(T)/2)
                        ret_α .+= (-2spinfac_uv .* spinfac_st
                                   .* Vσσ[tσ(s, t), tσ(u, v), :, :, :]
                                   .* ∇ρ[t, :, :, :, α] .* δσ[u, v][:, :, :])
                    end  # v
                end  # u
            end  # t
            ret_α
        end  # α
    end

    δV
end

function mergesum(nt1::NamedTuple{An}, nt2::NamedTuple{Bn}) where {An, Bn}
    all_keys = nothing
    ChainRulesCore.@ignore_derivatives begin
        all_keys = (union(An, Bn)..., )
    end
    values = map(all_keys) do key
        if haskey(nt1, key)
            nt1[key] .+ get(nt2, key, false)
        else
            nt2[key]
        end
    end
    NamedTuple{all_keys}(values)
end

_matify(::Nothing) = nothing
_matify(data::AbstractArray) = reshape(data, size(data, 1), :)

for fun in (:potential_terms, :kernel_terms)
    @eval begin
        function DftFunctionals.$fun(xc::Functional, density::LibxcDensities)
            $fun(xc, _matify(density.ρ_real), _matify(density.σ_real),
                     _matify(density.τ_real), _matify(density.Δρ_real))
        end

        function DftFunctionals.$fun(xcs::Vector{Functional}, density::LibxcDensities)
            isempty(xcs) && return NamedTuple()
            result = $fun(xcs[1], density)
            for i = 2:length(xcs)
                result = mergesum(result, $fun(xcs[i], density))
            end
            result
        end
    end
end


"""
Compute divergence of an operand function, which returns the Cartesian x,y,z
components in real space when called with the arguments 1 to 3.
The divergence is also returned as a real-space array.
"""
function divergence_real(operand, basis)
    gradsum = sum(1:3) do α
        operand_α = fft(basis, operand(α))
        map(G_vectors_cart(basis), operand_α) do G, operand_αG
            im * G[α] * operand_αG  # ∇_α * operand_α
        end
    end
    irfft(basis, gradsum)
end
