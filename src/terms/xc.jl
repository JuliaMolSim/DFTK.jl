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
@deprecate Xc(functional; kwargs...) Xc([functional]; kwargs...)

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
    τcore = nothing
    if (   xc.use_nlcc && any(needs_τ, xc.functionals)
        && any(has_core_kinetic_energy_density, basis.model.atoms))
        τcore = ρ_from_total(basis, atomic_total_density(basis, CoreKineticEnergyDensity()))
        minimum(τcore) < -sqrt(eps(T)) && @warn("Negative τcore detected: $(minimum(τcore))")
    end
    functionals = map(xc.functionals) do fun
        # Strip duals from functional parameters if needed
        params = parameters(fun)
        if !isempty(params)
            newparams = map(p -> convert_dual(T, p), params)
            fun = change_parameters(fun, newparams; keep_identifier=true)
        end
        fun
    end
    TermXc(convert(Vector{Functional}, functionals),
           convert_dual(T, xc.scaling_factor),
           T(xc.potential_threshold), ρcore, τcore)
end

function hybrid_parameters(xc::Xc)
    res = filter(!isnothing, map(hybrid_parameters, xc.functionals))
    isempty(res) ? nothing : only(res)
end

struct TermXc{T,CT,TCT} <: TermNonlinear where {T,CT,TCT}
    functionals::Vector{Functional}
    scaling_factor::T
    potential_threshold::T
    ρcore::CT
    τcore::TCT
end
DftFunctionals.needs_τ(term::TermXc) = any(needs_τ, term.functionals)

function xc_potential_real(term::TermXc, basis::PlaneWaveBasis{T}, ψ, occupation;
                           ρ, τ=nothing) where {T}
    @assert !isempty(term.functionals)
    @assert all(family(xc) in (:lda, :gga, :mgga, :mggal) for xc in term.functionals)

    if isnothing(τ) && needs_τ(term)
        throw(ArgumentError("TermXc needs the kinetic energy density τ. Please pass a `τ` " *
                            "keyword argument to your `Hamiltonian` or `energy_hamiltonian` call."))
    end

    # Add the model core charge density (non-linear core correction)
    if !isnothing(term.ρcore)
        ρ = ρ + term.ρcore
    end
    if !isnothing(term.τcore)
        τ = τ + term.τcore
    end

    max_ρ_derivs = maximum(max_required_derivative, term.functionals)
    density = LibxcDensities(basis, max_ρ_derivs, ρ, τ)
    _check_negative_bonding_indicator_α(density)

    n_spin = basis.model.n_spin_components
    potential_threshold = term.potential_threshold

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

@views @timing "ene_ops: xc" function ene_ops(term::TermXc, basis::PlaneWaveBasis,
                                              ψ, occupation; ρ, τ=nothing, kwargs...)
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

@views @timing "energy: xc"  function energy(term::TermXc, basis::PlaneWaveBasis{T},
                                             ψ, occupation; ρ, τ=nothing, kwargs...) where {T}
    if isnothing(τ) && needs_τ(term)
        throw(ArgumentError("TermXc needs the kinetic energy density τ. Please pass a `τ` " *
                            "keyword argument to your `energy` call."))
    end

    # Add the model core charge density (non-linear core correction)
    if !isnothing(term.ρcore)
        ρ = ρ + term.ρcore
    end
    if !isnothing(term.τcore)
        τ = τ + term.τcore
    end

    max_ρ_derivs = maximum(max_required_derivative, term.functionals)
    densities = LibxcDensities(basis, max_ρ_derivs, ρ, τ)
    _check_negative_bonding_indicator_α(densities)

    edensity = energy_density(term.functionals, densities)
    term.scaling_factor * sum(edensity) * basis.dvol
end

@timing "forces: xc" function compute_forces(term::TermXc, basis::PlaneWaveBasis{T},
                                             ψ, occupation; ρ, τ=nothing,
                                             kwargs...) where {T}
    # The only non-zero force contribution is from the nlcc core charge:
    # early return if nlcc is disabled / no elements have model core charges.
    isnothing(term.ρcore) && isnothing(term.τcore) && return nothing

    model = basis.model
    _, Vρ_real, Vτ_real = xc_potential_real(term, basis, ψ, occupation; ρ, τ)
    Vτ_fourier = nothing
    if model.spin_polarization in (:none, :spinless)
        Vρ_fourier = fft(basis, Vρ_real[:,:,:,1])
        if !isnothing(Vτ_real)
            Vτ_fourier = fft(basis, Vτ_real[:,:,:,1])
        end
    else
        Vρ_fourier = fft(basis, mean(Vρ_real, dims=4))
        if !isnothing(Vτ_real)
            Vτ_fourier = fft(basis, mean(Vτ_real, dims=4))
        end
    end

    forces_ρ = let
        form_factors, iG2ifnorm = atomic_density_form_factors(basis, CoreDensity())
        nlcc_groups = findall(group -> has_core_density(model.atoms[first(group)]),
                            model.atom_groups)

        _forces_xc(basis, Vρ_fourier, form_factors[:, nlcc_groups], iG2ifnorm,
                model.atom_groups[nlcc_groups])
    end
    if isnothing(Vτ_fourier)
        return forces_ρ
    end
    forces_τ = let
        form_factors, iG2ifnorm = atomic_density_form_factors(basis, CoreKineticEnergyDensity())
        nlcc_groups = findall(group -> has_core_kinetic_energy_density(model.atoms[first(group)]),
                            model.atom_groups)

        _forces_xc(basis, Vτ_fourier, form_factors[:, nlcc_groups], iG2ifnorm,
                model.atom_groups[nlcc_groups])
    end
    forces_ρ + forces_τ
end

# Function barrier to work around various type instabilities.
function _forces_xc(basis::PlaneWaveBasis{T}, Vxc_fourier::AbstractArray{U}, 
                    form_factors, iG2ifnorm, groups) where {T, U}
    # Pre-allocation of large arrays for GPU Efficiency
    TT = promote_type(T, real(U))
    Gs = G_vectors(basis)
    indices = to_device(basis.architecture, collect(1:length(Gs)))
    work = zeros_like(indices, Complex{TT}, length(indices))

    forces = Vec3{TT}[zero(Vec3{TT}) for _ = 1:length(basis.model.positions)]
    for (igroup, group) in enumerate(groups)
        for iatom in group
            r = basis.model.positions[iatom]
            ff_group = @view form_factors[:, igroup]
            map!(work, indices) do iG
                cis2pi(-dot(Gs[iG], r)) * conj(Vxc_fourier[iG]) * ff_group[iG2ifnorm[iG]]
            end

            forces[iatom] += map(1:3) do α
                tmp = sum(indices) do iG
                    -2π*im*Gs[iG][α] * work[iG]
                end
                -real(tmp / sqrt(basis.model.unit_cell_volume))
            end
        end
    end
    forces
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
see also Richard Martin, Electronic structure, p. 158. For meta-GGAs an extra term ΔVl appears
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
struct LibxcDensities{T}
    basis::PlaneWaveBasis{T}
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
function LibxcDensities(basis::PlaneWaveBasis{T}, max_derivative::Integer, ρ, τ) where {T}
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

        # Spin index transformation (s, t) => st as expected by Libxc
        tσ = DftFunctionals.spinindex_σ
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
    LibxcDensities{T}(basis, max_derivative, ρ_real, ∇ρ_real, σ_real, Δρ_real, τ_Libxc)
end

function _check_negative_bonding_indicator_α(densities::LibxcDensities{T}) where {T}
    if !isnothing(densities.τ_real) && !isnothing(densities.σ_real)
        n_spin = densities.basis.model.n_spin_components
        has_negative_α = @views any(1:n_spin) do iσ
            # α = (τ - τ_W) / τ_unif should be positive with τ_W = |∇ρ|² / 8ρ
            # equivalently, check 8ρτ - |∇ρ|² ≥ 0
            α_check = (8 .* densities.ρ_real[iσ, :, :, :] .* densities.τ_real[iσ, :, :, :]
                       .- densities.σ_real[DftFunctionals.spinindex_σ(iσ, iσ), :, :, :])
            any(α_check .<= -sqrt(eps(T)))
        end
        if has_negative_α
            @warn "Exchange-correlation term: the kinetic energy density τ is smaller " *
                  "than the von Weizsäcker kinetic energy density τ_W somewhere. " *
                  "This can lead to unphysical results. " *
                  "This can be caused by pseudopotentials without a non-linear core correction " *
                  "for τ, or by an unphysical initial guess for τ. " *
                  "This message is only logged once." maxlog=1
        end
    end
end

function compute_kernel(term::TermXc, basis::PlaneWaveBasis{T}; ρ, kwargs...) where {T}
    n_spin  = basis.model.n_spin_components
    @assert 1 ≤ n_spin ≤ 2
    if !all(family(xc) == :lda for xc in term.functionals)
        error("compute_kernel only implemented for LDA")
    end

    # For LDA the Kernel is known to be diagonal, so we can get away
    # with a single push-forward (two for spin-polarized case)
    if n_spin == 1
        f_spinless(ε) = xc_potential_real(term, basis, nothing, nothing; ρ=ρ.+ε).potential
        δpotential = ForwardDiff.derivative(f_spinless, zero(T))
        Diagonal(vec(δpotential))
    else
        # We could use chunking instead, but this is simpler and not performance-critical.
        function f_collinear(ε)
            dρ1 = reshape([ε, 0], 1, 1, 1, 2)
            dρ2 = reshape([0, ε], 1, 1, 1, 2)
            stack([xc_potential_real(term, basis, nothing, nothing; ρ=ρ.+dρ1).potential,
                   xc_potential_real(term, basis, nothing, nothing; ρ=ρ.+dρ2).potential])
        end
        δpotential = ForwardDiff.derivative(f_collinear, zero(T))

        # Blocks in the kernel matrix mapping (ρα, ρβ) ↦ (Vα, Vβ)
        Kαα = @view δpotential[:, :, :, 1, 1]
        Kαβ = @view δpotential[:, :, :, 1, 2]
        Kβα = @view δpotential[:, :, :, 2, 1]
        Kββ = @view δpotential[:, :, :, 2, 2]
        [Diagonal(vec(Kαα)) Diagonal(vec(Kαβ));
         Diagonal(vec(Kβα)) Diagonal(vec(Kββ))]
    end
end


function apply_kernel(term::TermXc, basis::PlaneWaveBasis{T}, δρ::AbstractArray{Tδρ};
                      ρ, q=zero(Vec3{T}), kwargs...) where {T, Tδρ<:Union{T,Complex{T}}}
    isempty(term.functionals) && return nothing
    @assert (all(family(xc) in (:lda, :gga, :mggal) && !needs_τ(xc) for xc in term.functionals))

    if !iszero(q) && !all(family(xc) == :lda for xc in term.functionals)
        error("Phonons are currently only implemented for LDA")
    end

    # Key insight: kernel application is just a Hessian-vector product,
    # which is computed with a push-forward of the gradient.
    f(ρ_eval) = xc_potential_real(term, basis, nothing, nothing; ρ=ρ_eval).potential
    Tag = typeof(ForwardDiff.Tag(f, T))
    if Tδρ <: T
        # Usually δρ has the same type, so we do a standard push-forward
        ε = Dual{Tag}(zero(T), one(T))
        ForwardDiff.partials.(f(ρ .+ ε .* δρ), 1)
    else
        # But for complex δρ (phonons) we need to push the real and imaginary
        # parts forward separately
        ε1 = Dual{Tag}(zero(T), one(T), zero(T))
        ε2 = Dual{Tag}(zero(T), zero(T), one(T))
        potential = f(ρ .+ ε1 .* real.(δρ) .+ ε2 .* imag.(δρ))
        ForwardDiff.partials.(potential, 1) .+ im .* ForwardDiff.partials.(potential, 2)
    end
end

@views function compute_dynmat(term::TermXc, basis::PlaneWaveBasis{T}, ψ, occupation;
                               ρ, δρs, q=zero(Vec3{T}), kwargs...) where {T}
    isnothing(term.ρcore) && return nothing

    # Hellmann-Feynman:
    # ∂E/∂u_tβ(q) = ∫ Vxc ∂ρcore/∂u_tβ(q)
    # Differentiate again wrt. u_sα(-q) to get the dynamical matrix element:
    # ∂²E/∂u_sα(-q)∂u_tβ(q) = ∫ Kxc (∂ρ/∂u_sα(-q) + ∂ρcore/∂u_sα(-q)) (∂ρcore/∂u_tβ(q))
    #                       + ∫ Vxc ∂²ρcore/∂u_sα(-q)∂u_tβ(q)

    n_atoms = length(basis.model.positions)
    n_dim = basis.model.n_dim
    δρcores = zero.(δρs)
    for s = 1:n_atoms, α = 1:n_dim
        δρcores[α, s] .= derivative_wrt_αs(basis.model.positions, α, s) do positions_αs
            ρ_from_total(basis, atomic_total_density(basis, CoreDensity();
                                                     q, positions=positions_αs))
        end
    end

    dynmat = zeros(complex(T), 3, n_atoms, 3, n_atoms)
    # Assemble first term directly
    for s = 1:n_atoms, α = 1:n_dim
        # We actually receive ∂ρ/∂u_sα(q) (note: +q and not -q),
        # however since the potential is real, we conjugate to obtain the -q term:
        # δV =       Kxc (∂ρ/∂u_sα(-q) + ∂ρcore/∂u_sα(-q))
        #    = conj( Kxc (∂ρ/∂u_sα( q) + ∂ρcore/∂u_sα( q)) )
        δV_αs = conj.(apply_kernel(term, basis, δρs[α, s] + δρcores[α, s]; ρ, q))
        for t = 1:n_atoms, β = 1:n_dim
            δρcore_tβ = δρcores[β, t]
            dynmat[β, t, α, s] = sum(δV_αs .* δρcore_tβ) * basis.dvol
        end
    end

    # For the second term ∫ Vxc ∂²ρcore/∂u_sα(-q)∂u_tβ(q),
    # we only have contributions for s == t.
    # Additionally the -q and +q phases cancel out.
    Vxc = xc_potential_real(term, basis, ψ, occupation; ρ).potential
    for s = 1:n_atoms, α = 1:n_dim, β = 1:n_dim
        δ²ρcore = derivative_wrt_αs(basis.model.positions, β, s) do positions_βs
            derivative_wrt_αs(positions_βs, α, s) do positions_βsαs
                ρ_from_total(basis, atomic_total_density(basis, CoreDensity();
                                                         positions=positions_βsαs))
            end
        end
        dynmat[β, s, α, s] += sum(Vxc .* δ²ρcore) * basis.dvol
    end

    dynmat
end

function compute_δHψ_αs(term::TermXc, basis::PlaneWaveBasis, ψ, α, s, q; ρ)
    isnothing(term.ρcore) && return nothing

    # With an NLCC, an atom displacement triggers a change in the XC potential.
    # We compute the change in the NLCC first, then we apply the kernel to get the
    # change in potential. Finally, we apply it to the wavefunctions.
    δρcore_αs = derivative_wrt_αs(basis.model.positions, α, s) do positions_αs
        ρ_from_total(basis, atomic_total_density(basis, CoreDensity();
                                                 q, positions=positions_αs))
    end
    δV_αs = apply_kernel(term, basis, δρcore_αs; ρ, q)
    multiply_ψ_by_blochwave(basis, ψ, δV_αs, q)
end

function mergesum(nt1::NamedTuple{An}, nt2::NamedTuple{Bn}) where {An, Bn}
    all_keys = (union(An, Bn)..., )
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

function DftFunctionals.potential_terms(xc::DispatchFunctional, density::LibxcDensities)
    potential_terms(xc, _matify(density.ρ_real), _matify(density.σ_real),
                        _matify(density.τ_real), _matify(density.Δρ_real))
end

# Ensure functionals from DftFunctionals are sent to the CPU
# TODO: Allow GPUArrys once DftFunctionals is refactored to support GPU. 
function DftFunctionals.potential_terms(fun::DftFunctionals.Functional, density::LibxcDensities)
    maticpuify(::Nothing) = nothing
    maticpuify(x::AbstractArray) = reshape(Array(x), size(x, 1), :)
    DftFunctionals.potential_terms(fun, maticpuify(density.ρ_real), maticpuify(density.σ_real),
                                        maticpuify(density.τ_real), maticpuify(density.Δρ_real))
end

function DftFunctionals.potential_terms(xcs::Vector{Functional}, density::LibxcDensities)
    isempty(xcs) && return NamedTuple()
    result = DftFunctionals.potential_terms(xcs[1], density)
    for i = 2:length(xcs)
        result = mergesum(result, DftFunctionals.potential_terms(xcs[i], density))
    end
    result
end

# Ensure functionals from DftFunctionals are sent to the CPU
# TODO: Allow GPUArrys once DftFunctionals is refactored to support GPU. 
function DftFunctionals.energy_density(fun::DftFunctionals.Functional, density::LibxcDensities)
    maticpuify(::Nothing) = nothing
    maticpuify(x::AbstractArray) = reshape(Array(x), size(x, 1), :)
    DftFunctionals.energy_density(fun, maticpuify(density.ρ_real), maticpuify(density.σ_real),
                                       maticpuify(density.τ_real), maticpuify(density.Δρ_real))
end
function DftFunctionals.energy_density(xc::DispatchFunctional, density::LibxcDensities)
    energy_density(xc, _matify(density.ρ_real), _matify(density.σ_real),
                       _matify(density.τ_real), _matify(density.Δρ_real))
end
function DftFunctionals.energy_density(xcs::Vector{Functional}, density::LibxcDensities{T}) where {T}
    xcs = filter(has_energy, xcs)
    isempty(xcs) && return zero(T)
    result = energy_density(xcs[1], density)
    for i = 2:length(xcs)
        result += energy_density(xcs[i], density)
    end
    result
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
