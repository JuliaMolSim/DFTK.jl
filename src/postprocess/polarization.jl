# Berry phase electronic polarization in direction α:
#   Pel,cart_α({uₙₖ}) = -2i/(2π)^3 ∑ₙ ∫ <uₙₖ|∂kcart_α uₙₖ>
#                    = -i/Ω * Σₙₖ wₖfₙₖ <uₙₖ|∂kcart_α uₙₖ>,
# where the derivative ∂kcart_α is computed in Cartesian coordinates.
# In reduced coordinates, k = 1/2π L^T kcart hence
#   Pel,cart_α({uₙₖ}) = ∑_β L_αβ / Ω * [ -i/2π Σₙₖ wₖfₙₖ <uₙₖ|∂k_α uₙₖ> ]
#
# To gracefully treat the combination of strain and electric fields,
# we write all internal DFTK calculations in terms of the
# "fully reduced" polarization discussed in the Appendix of
# [Wu et al., Phys. Rev. B 72, 035105 (2005)].
# The reduced electronic polarization is:
#   Pel_α({uₙₖ}) = -i/2π Σₙₖ wₖfₙₖ <uₙₖ|∂k_α uₙₖ>
# and the reduced ionic polarization is: (Rₛ in reduced coordinates)
#   Pion_α = ∑ₛ Zₛ Rₛ_α.
# The Cartesian polarization is:
#   Pcart = L/Ω P
#
# At nonzero field, the {uₙₖ} minimize the electric enthalpy functional:
#   F({uₙₖ}, E) = Energy({uₙₖ}) - E ⋅ Ptot({uₙₖ}).
# The reduced polarization Ptot is defined above and the
# reduced electric field E is defined such that Ecart = L⁻ᵀ E.
# It holds that E ⋅ Ptot = Ω Ecart ⋅ Pcart; the latter is the usual
# literature definition, but the reduced variants are more natural when
# combining electric fields and strain perturbations.
#
# # Analytic k-derivative
# The k-derivative can be evaluated analytically, which allows perturbations
# to be computed at zero field.
# The DFPT rhs for an electric field perturbation in direction α is
# 1/wₖfₙₖ ∂²F/∂<uₙₖ|∂E_α = -1/wₖfₙₖ ∂Pel_α/∂<uₙₖ| = i/2π ∂k_α uₙₖ.
# So the steps are:
# 1. Compute ∂k_α|uₙₖ> as response to ∂H/∂k_α (Sternheimer).
# 2. Compute DFPT response to i/2π ∂k_α uₙₖ.
# 3. Evaluate required derivative of quantity wrt. uₙₖ with AD
#
# For the special case of a polarization derivative ∂P_α/∂θ, step 3 is replaced by:
# Pel_α({uₙₖ}) = -i/2π Σₙₖ wₖfₙₖ <uₙₖ|∂k_α uₙₖ>
# ∂Pel_α({uₙₖ})/∂θ = -i/2π * Σₙₖ wₖfₙₖ (<∂θ uₙₖ|∂k_α uₙₖ> + <uₙₖ|∂k_α∂θ uₙₖ>)
#                 (by parts wrt k because the sum approximates an integral over k)
#                 = -i/2π * Σₙₖ wₖfₙₖ (<∂θ uₙₖ|∂k_α uₙₖ> - <∂k_α uₙₖ|∂θ uₙₖ>)
#                 = -i/2π * Σₙₖ wₖfₙₖ (<∂θ uₙₖ|∂k_α uₙₖ> - conj(<∂ θuₙₖ|∂k_α uₙₖ>))
#                 =  1/2π * Σₙₖ wₖfₙₖ 2 Im <∂θ uₙₖ|∂k_α uₙₖ>.
#
# # Numerical k-derivative
# The k-derivative can be evaluated numerically with finite differences,
# which allows both solving for the metastable "ground state" at a finite field,
# and the response to perturbations (perturbation expansion after discretization aka. PEAD)
# TODO: implement numerical derivative (Berry phase formula) in DFTK

abstract type PolarizationMethod end
@kwdef struct LinearResponsePolarization <: PolarizationMethod
    verbose::Bool = true
    # TODO: not sure this is the best place to specify it?
    tol::Float64 = 1e-10
end

function electronic_polarization(basis, ψ, occupation, ::LinearResponsePolarization;
                                 kwargs...)
    @warn "electronic_polarization(::LinearResponsePolarization) always returns zero"
    zero(Vec3{eltype(basis)})
end
@timing function electronic_polarization(basis::PlaneWaveBasis{T}, ψ, occupation,
                                 lrpol::LinearResponsePolarization;
                                 εF, eigenvalues, occupation_threshold,
                                 ρ=nothing, τ=nothing) where {Tg,V,T<:Dual{Tg,V}}
    basis0 = construct_value(basis)
    ψ0           = map(ψ) do ψk
        ForwardDiff.value.(Tg, real.(ψk)) .+ im .* ForwardDiff.value.(Tg, imag.(ψk))
    end
    occupation0  = [ForwardDiff.value.(Tg, occk) for occk in occupation]
    εF0          = ForwardDiff.value(Tg, εF)
    eigenvalues0 = [ForwardDiff.value.(Tg, εk) for εk in eigenvalues]
    ρ0           = ForwardDiff.value.(Tg, ρ)
    τ0           = isnothing(τ) ? nothing : ForwardDiff.value.(Tg, τ)

    # The polarization is only well-defined for insulators!
    filled_occ = filled_occupation(basis0.model)
    for occk in occupation0
        all(abs.(occk) .<= occupation_threshold .|| abs.(occk .- filled_occ) .<= occupation_threshold) ||
            error("Only full or zero occupations are supported for polarization, found partial occupation: $occk.")
    end

    ham = Hamiltonian(basis0; ψ=ψ0, occupation=occupation0, ρ=ρ0, τ=τ0)
    bandtolalg = BandtolBalanced(basis0, ψ0, occupation0; occupation_threshold)

    dψ_dk = map(1:3) do α
        lrpol.verbose && mpi_master(basis0.comm_kpts) && println("Computing dunk/dk_$α")
        δk = zeros(V, 3)
        δk[α] = one(V)
        dH_dkψ = compute_δHψδk(basis0, ψ0, δk; ρ=ρ0, τ=τ0)
        apply_χ0_4P(ham, ψ0, occupation0, εF0, eigenvalues0, dH_dkψ;
                    occupation_threshold, bandtolalg, lrpol.tol).δψ
    end
    P = zero(Vec3{V})
    dP = ntuple(ForwardDiff.npartials(T)) do n
        δψ = [ForwardDiff.partials.(Tg, ψk, n) for ψk in ψ]
        δP = Vec3{V}(map(1:3) do α
            δPk = map(occupation0, δψ, dψ_dk[α]) do occk, δψk, dψk_dk
                1/V(π) * sum(occk .* imag.(columnwise_dots(δψk, dψk_dk)))
            end
            weighted_ksum(basis0, δPk)
        end)
        # TODO: missing symmetrization
        δP
    end
    map(P, dP...) do Pα, dPα...
        ForwardDiff.Dual{Tg}(Pα, dPα)
    end
end

function electronic_polarization(scfres, method::PolarizationMethod)
    electronic_polarization(scfres.basis, scfres.ψ, scfres.occupation, method;
                            scfres.εF, scfres.eigenvalues, scfres.occupation_threshold,
                            scfres.ρ, scfres.τ)
end

function ionic_polarization(model::Model)
    sum(zip(model.positions, model.atoms)) do (pos, atom)
        pos * charge_ionic(atom)
    end
end

function total_polarization(basis::PlaneWaveBasis, ψ, occupation,
                            method::PolarizationMethod; kwargs...)
    ( electronic_polarization(basis, ψ, occupation, method; kwargs...)
    + ionic_polarization(basis.model))
end
function total_polarization(scfres, method::PolarizationMethod)
    electronic_polarization(scfres, method) + ionic_polarization(scfres.basis.model)
end

function total_polarization_cart(basis::PlaneWaveBasis, ψ, occupation,
                                 method::PolarizationMethod; kwargs...)
    model = scfres.basis.model
    1/model.unit_cell_volume * model.lattice * total_polarization(basis, ψ, occupation,
                                                                  method; kwargs...)
end
function total_polarization_cart(scfres, method::PolarizationMethod)
    model = scfres.basis.model
    1/model.unit_cell_volume * model.lattice * total_polarization(scfres, method)
end

@doc raw"""
Compute the Jacobian-vector product of H_k u_{nk} wrt. k in direction δk.
The passed δk must be in reduced coordinates!
"""
function compute_δHψδk(basis::PlaneWaveBasis, ψ, δk; ρ=nothing, τ=nothing)
    function apply_kshift(ε::T) where {T}
        kcoords = map(basis.kcoords_global) do kcoord
            kcoord + ε * δk
        end
        kgrid_shifted = ExplicitKpoints(kcoords, convert.(T, basis.kweights))
        model = Model(basis.model; lattice=convert.(T, basis.model.lattice))
        basis_shifted = PlaneWaveBasis(model; basis.Ecut, kgrid=kgrid_shifted,
                                       basis.variational, basis.fft_size,
                                       basis.symmetries_respect_rgrid,
                                       basis.use_symmetries_for_kpoint_reduction,
                                       basis.comm_kpts, basis.architecture)
        # TODO: what would happen if we were to recompute ρ and τ?
        #       in principle the k derivatives in τ should not change the result, should test this!
        ρ_new = isnothing(ρ) ? nothing : convert.(T, ρ)
        τ_new = isnothing(τ) ? nothing : convert.(T, τ)
        ham = Hamiltonian(basis_shifted; ρ=ρ_new, τ=τ_new)
        ham * ψ
    end
    ForwardDiff.derivative(apply_kshift, zero(eltype(basis)))
end

function compute_ψbars(ham, ψ, occupation, εF, eigenvalues;
                       ρ=nothing, τ=nothing, occupation_threshold, kwargs...)
    model = ham.basis.model
    if length(model.symmetries) > 1
        error("Symmetry handling not yet implemented for ψbar computations.")
    end
    # Full occupations are mandatory for the ψbar construction via Sternheimer
    filled_occ = filled_occupation(model)
    for occk in occupation
        all(abs.(occk) .<= occupation_threshold .|| abs.(occk .- filled_occ) .<= occupation_threshold) ||
            error("Only full or zero occupations are supported for ψbar computations, found partial occupation: $occk.")
    end

    δHδk_ψ = compute_δHδk(ham.basis, ψ; ρ, τ)
    map(δHδk_ψ) do δHδkα_ψ
        χ0_res = apply_χ0_4P(ham, ψ, occupation, εF, eigenvalues, im * δHδkα_ψ;
                             occupation_threshold, kwargs...)
        ψbar = χ0_res.δψ
        χ0_res = Base.structdiff(χ0_res, NamedTuple{(:δψ,)}) # remove δψ from res tuple
        (; ψbar, χ0_res...)
    end
end

function electric_field_responses(ham, ρ, ψ, occupation, εF, eigenvalues;
                                  τ=nothing,
                                  tol=1e-8,
                                  occupation_threshold,
                                  bandtolalg=BandtolBalanced(ham.basis, ψ, occupation; occupation_threshold),
                                  factor_ψbar=1/10,
                                  kwargs...)
    ψbars = compute_ψbars(ham, ψ, occupation, εF, eigenvalues;
                          ρ, τ, occupation_threshold,
                          bandtolalg, tol=tol*factor_ψbar)
    electric_field_responses = map(ψbars) do ψbar_α
        solve_ΩplusK_split(ham, ρ, ψ, occupation, εF, eigenvalues, ψbar_α.ψbar;
                           tol, occupation_threshold, bandtolalg, τ, kwargs...)
    end
    (; ψbars, electric_field_responses)
end

function electric_field_responses(scfres::NamedTuple; kwargs...)
    electric_field_responses(scfres.ham, scfres.ρ, scfres.ψ, scfres.occupation,
                             scfres.εF, scfres.eigenvalues;
                             scfres.τ, scfres.occupation_threshold,
                             bandtolalg=BandtolBalanced(scfres), kwargs...)
end

"""
Compute the dielectric tensor ϵ_αβ = ∂D_α/∂E_β in Cartesian coordinates,
where E is an applied electric field and D is the resulting electric displacement field.
"""
function compute_dielectric_cart(basis, occupation; ψbars, electric_field_responses)
    model = basis.model
    Ω = model.unit_cell_volume
    χ = zeros(eltype(basis), 3, 3)
    for β in 1:3, α in 1:3
        # χ_αβ = ∂P_α/∂E_β in reduced coordinates
        δψ_β = electric_field_responses[β].δψ
        χ[α, β] = weighted_ksum(basis, map(ψbars[α].ψbar, δψ_β, occupation) do ψkbar, δψk, occk
            -2/Ω * sum(occk .* real.(columnwise_dots(ψkbar, δψk)))
        end)
    end
    # The ddk perturbation is computed in reduced coordinates,
    # so we have to multiply by A / 2π on the left and A^T / 2π on the right
    # to convert χ to Cartesian. The vacuum δ_αβ term must be added separately
    # (not transformed), since it is already in Cartesian.
    χ_cart = model.lattice * χ * model.lattice' / (2π)^2
    # ϵ_αβ = δ_αβ + 4π χ_αβ
    I + 4π * χ_cart
end

function compute_dielectric_cart(scfres::NamedTuple; ψbars, electric_field_responses)
    compute_dielectric_cart(scfres.basis, scfres.occupation;
                            ψbars, electric_field_responses)
end

"""
Compute the Born effective charge tensor Z*.
The indices are Z*[α, β, s] = Ω ∂P_α/∂u_sβ,
i.e. the change in total polarization in Cartesian direction α
due to a displacement of atom s in Cartesian direction β.

Here we use the equivalent (via the electric enthalpy) definition
Z*[α, β, s] = ∂F_sβ/∂E_α where F is the force,
which only requires the 3 electric field responses.
"""
function compute_Zstar_cart(basis, ρ, ψ, occupation; τ=nothing, electric_field_responses)
    atoms = basis.model.atoms
    n_atoms = length(atoms)

    T = eltype(basis)
    Zstar = zeros(T, 3, 3, n_atoms)
    for α in 1:3
        resp = electric_field_responses[α]
        f(ε) = compute_forces_cart(basis, ψ .+ ε.*resp.δψ, occupation;
                                   ρ=ρ .+ ε.*resp.δρ,
                                   τ=isnothing(resp.δτ) ? nothing : τ .+ ε.*resp.δτ)
        Zstar[α, :, :] = stack(ForwardDiff.derivative(f, zero(T)))
    end
    # convert ddk perturbation to Cartesian coordinates
    Zstar_cart = zero(Zstar)
    for s in 1:n_atoms, β in 1:3
        Zstar_cart[:, β, s] = basis.model.lattice * Zstar[:, β, s] / 2π
    end
    # finally we add the ionic contribution
    for s in 1:n_atoms
        Zstar_cart[:, :, s] += I * charge_ionic(atoms[s])
    end
    Zstar_cart
end

function compute_Zstar_cart(scfres::NamedTuple; electric_field_responses)
    compute_Zstar_cart(scfres.basis, scfres.ρ, scfres.ψ, scfres.occupation;
                       scfres.τ, electric_field_responses)
end
