# TODO: names need another pass :)
# TODO: careful with minus signs

# Electric field perturbations at zero field.
#
# We start from the electric enthalpy functional
# F({uₙₖ}, E) = Energy({uₙₖ}) - Ω E * Ptot({uₙₖ}),
# where Ptot is the total polarization (electronic + ionic).
# Its electronic part in direction α is given by
# Pel_α({uₙₖ}) = - Σₙₖ wₖfₙₖ <uₙₖ|i ∂k_α|uₙₖ>.
#
# The first step of the computation is to evaluate i ∂k_α|uₙₖ>,
# which requires only a bare χ0 solve per direction α.
# For an electric field perturbation, the perturbing potential is:
# 1/fₙₖ ∂²F/∂uₙₖ*∂E = -Ω/fₙₖ ∂P/∂uₙₖ* = Ω i ∂k_α|uₙₖ>,
# and we have to solve the full self-consistent DFPT equation.
#
#
# The fundamental trick is that while the position operator r_α is ill-defined in
# a crystal, for m != n and εₘₖ != εₙₖ, we can compute
# <ψₘₖ|r_α|ψₙₖ> = <ψₘₖ|[H, r_α]|ψₙₖ> / (εₘₖ - εₙₖ)
#             = 1/im * <ψₘₖ|∂H/∂k_α|ψₙₖ> / (εₘₖ - εₙₖ),
# which is well-defined.
# Assuming a non-zero band gap and full occupations,
# the Sternheimer equation, given a δH, gives us δψₙₖ:
# |δψₙₖ> = Σₘ <ψₘₖ|δH|ψₙₖ> / (εₘₖ - εₙₖ) |ψₘₖ>
# where the sum runs over unoccupied states m.
# By solving a Sternheimer equation given -1/im * ∂H/∂k_α|ψₙₖ> as RHS,
# we thus obtain an effective "r_α|ψₙₖ>", which we call `ψbar_α`.
#
# To compute the response to an electric field change "V = r_α δE_α" (for electrons),
# we compute the self-consistent response to ψbar_α.
#
# Similarly, the polarization change
# δP_α = " -1/Ω ∫ δρ(r) r_α dr "
# is rewritten as:
#      = -4/Ω Re Σₙₖ <ψₙₖ|r_α|δψₙₖ>
#      = -4/Ω Re Σₙₖ <ψbar_αₙₖ|δψₙₖ>

# TODO: it might be cleaner to start from the electric enthalpy
# F(ψ, E) = Energy(Ψ) - Ω E * P(Ψ)
# and motivative the ψbar construction from the implicit function derivative
# inv(δ²Energy/δψ²) * ∂²F/δψδE,
# where ∂²F/δψδE ends up being precisely ψbar.

function compute_δHδk(basis::PlaneWaveBasis{T}, ψ; ρ=nothing, τ=nothing) where {T}
    model = basis.model
    function apply_kshift(kshift)
        T_new = promote_type(eltype(basis), eltype(kshift))
        # TODO: likely not mpi compatible yet
        kcoords = map(basis.kpoints) do kpt
            kpt.coordinate + kshift
        end
        kpoints_shifted = ExplicitKpoints(kcoords, convert.(T_new, basis.kweights))
        model_new = Model(model; lattice=convert.(T_new, model.lattice))
        basis_shifted = PlaneWaveBasis(model_new; Ecut=basis.Ecut, kgrid=kpoints_shifted)
        ρ_new = isnothing(ρ) ? nothing : convert.(T_new, ρ)
        τ_new = isnothing(τ) ? nothing : convert.(T_new, τ)
        ham = Hamiltonian(basis_shifted; ρ=ρ_new, τ=τ_new)
        ham * ψ
    end
    map(1:3) do α
        kshift = zeros(T, 3)
        kshift[α] = one(T)
        ForwardDiff.derivative(ε -> apply_kshift(ε .* kshift), zero(T))
    end
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
