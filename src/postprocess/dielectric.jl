# Homogeneous electric-field response of an insulator: the electronic (clamped-ion)
# dielectric tensor ε∞ and the static polarizability, from density-functional perturbation
# theory. See DIELECTRIC_RESPONSE_PLAN.md and Baroni et al., Rev. Mod. Phys. 73, 515 (2001),
# §II.C (the "d/dk" trick of the modern theory of polarization).
#
# The perturbing potential of a uniform field is E·r, but the position operator r is
# ill-defined under periodic boundary conditions. The fix is to work with the well-defined
# commutator [H, r_α] = i ∂H_k/∂k_α (the velocity operator), which gives the projected-position
# orbitals ψ̄^α = P_c r_α ψ by one non-interacting Sternheimer solve, followed by the usual
# self-consistent (Ω+K) response. Both reuse the existing DFPT machinery at q = 0.

@doc raw"""
Apply the velocity operator (position commutator) to the orbitals, for each Cartesian
direction ``α = 1:3``:
```math
[H_k, r_α] ψ_k = i\, \frac{∂H_k}{∂k_α} ψ_k.
```
The derivative w.r.t. the Cartesian crystal momentum is taken by ForwardDiff: [`shift_kpoints`](@ref)
rebuilds the basis at a `Dual`-shifted k-point coordinate, the (shifted) Hamiltonian is applied to
the fixed orbitals `ψ`, and the `t`-partial of the result is extracted. The kinetic multiplier and
the nonlocal projectors carry the k-dependence (the latter is the "nonlocal commutator"); the
local/Hartree/xc terms are k-independent and drop out. `δHψ[α][ik]` is ``[H_k, r_α] ψ_{ik}`` on the
G-sphere of `basis.kpoints[ik]`.
"""
function compute_δHψ_dk(basis::PlaneWaveBasis{T}, ψ, ρ) where {T}
    # A unit Cartesian displacement of k is the reduced shift recip_lattice⁻¹ eα, since
    # k_red = recip_lattice⁻¹ k_cart; differentiating H along it gives ∂H/∂k_cart,α.
    Binv = inv(basis.model.recip_lattice)
    tag = ForwardDiff.Tag(compute_δHψ_dk, T)
    map(1:3) do α
        # Shift all k-points by t·(reduced eα) with t a Dual seed, and apply the shifted Hamiltonian
        # to the fixed orbitals. The Dual enters only through the kinetic/nonlocal operators.
        t = ForwardDiff.Dual{typeof(tag)}(zero(T), one(T))
        ham_t = Hamiltonian(shift_kpoints(basis, t .* Vec3(Binv[:, α])); ρ)
        map(1:length(basis.kpoints)) do ik
            Hψk = ham_t[ik] * ψ[ik]              # value = H_k ψ, partial = (∂H_k/∂k_α) ψ
            im .* ForwardDiff.partials.(Hψk, 1)  # [H_k, r_α] ψ = i (∂H_k/∂k_α) ψ
        end
    end
end

# The d/dk formula captures only the interband (bound) response, so it is valid for insulators.
# Reject fractional occupations (a metal, or T>0 smearing); empty conduction bands are fine.
function _assert_insulator(basis::PlaneWaveBasis, occupation, occupation_threshold)
    filled = filled_occupation(basis.model)
    for occ_k in occupation, occ in occ_k
        empty = occ < occupation_threshold
        full  = occ > filled - occupation_threshold
        empty || full || error(
            "Electric-field response is implemented for insulators; the occupation $occ_k " *
            "has a fractional entry, so the metallic intraband term (not included) would " *
            "dominate. See DIELECTRIC_RESPONSE_PLAN.md §6.")
    end
end

# Contract ε∞[β,α] = -(8π/Ω) Σ_k w_k Σ_n f_n Re⟨ψ̄^β | δψ^α⟩ (without the prefactor and sign);
# returns Σ_k w_k Σ_n f_n Re⟨ψ̄^β_nk | δψ^α_nk⟩.
function _dielectric_contraction(basis::PlaneWaveBasis, ψ̄β, δψα, occupation)
    per_k = map(enumerate(basis.kpoints)) do (ik, kpt)
        sum(occupation[ik][n] * real(dot(ψ̄β[ik][:, n], δψα[ik][:, n]))
            for n = 1:size(δψα[ik], 2); init=zero(real(eltype(basis))))
    end
    weighted_ksum(basis, per_k)
end

@doc raw"""
Compute the electronic (clamped-ion) **dielectric tensor** ``ε∞`` and the static
**polarizability** of an insulator from its response to a homogeneous electric field
(density-functional perturbation theory). Returns a NamedTuple
`(; ε∞, polarizability, δψ, ψ̄)` where
```math
ε∞[β,α] = δ_{βα} - \frac{8π}{Ω} ∑_k w_k ∑_n f_n \operatorname{Re}⟨ψ̄^β_{nk} | δψ^α_{nk}⟩,
```
`polarizability` `= Ω·χ` is the per-cell polarizability (``P = χ·E``), `δψ[α]` is the
self-consistent orbital response to a unit field along Cartesian direction `α`, and `ψ̄[α]` are
the projected-position orbitals ``P_c r_α ψ``.

!!! warning "Preliminary implementation — insulators only"
    Only the interband (bound) response is included, so this is valid for **insulators**; a
    metal's intraband/Drude term is missing (it errors on fractional occupations).
    Spin-unpolarized only for now. The k-point symmetry reduction cannot be
    reused directly (it would over-symmetrize the field response to zero), so the basis must
    carry no symmetries: build it from `Model(system; symmetries=false)`. A stabilizer-subgroup
    reduction is planned; see `DIELECTRIC_RESPONSE_PLAN.md`.
"""
@timing function compute_dielectric(scfres::NamedTuple; kwargs...)
    basis = scfres.basis
    T = eltype(basis)

    # The projected-position / d/dk formula gives only the interband response: insulators only.
    _assert_insulator(basis, scfres.occupation, scfres.occupation_threshold)
    # v1 is spin-unpolarized only (the contraction sums spin channels, but this is untested).
    basis.model.n_spin_components == 1 || error(
        "compute_dielectric currently only supports spin-unpolarized (`:none`) models.")
    # The q=0 response solver assumes time-reversal symmetry, like the phonon code.
    @assert !any(breaks_time_reversal_symmetry, basis.model.term_types) (
        "The electric-field response requires time-reversal symmetry.")
    # Meta-GGA adds a k-dependent DivAgrad operator to the XC term whose velocity contribution
    # is not yet wired into `apply_kblock`; refuse rather than silently drop it.
    !any(needs_τ, basis.terms) || error(
        "compute_dielectric does not yet support meta-GGA (τ-dependent) functionals.")
    # The field response breaks the k-point symmetries; a symmetry-reduced basis would
    # over-symmetrize δρ^α to zero (see DIELECTRIC_RESPONSE_PLAN.md §7).
    length(basis.symmetries) == 1 || error(
        "compute_dielectric requires a symmetry-free basis; build the model with " *
        "`Model(system; symmetries=false)`.")

    Ω = basis.model.unit_cell_volume
    q = zero(Vec3{T})

    # Velocity RHS [H, r_α]ψ, then ψ̄^α = P_c r_α ψ by one non-interacting Sternheimer solve.
    δHψ = compute_δHψ_dk(basis, scfres.ψ, scfres.ρ)
    ψ̄ = map(1:3) do α
        apply_χ0_4P(scfres.ham, scfres.ψ, scfres.occupation, scfres.εF, scfres.eigenvalues,
                    δHψ[α]; q, scfres.occupation_threshold,
                    bandtolalg=BandtolBalanced(scfres)).δψ
    end

    # δψ^α: full self-consistent (Ω+K) response to the field along α, then contract.
    ε∞ = zeros(T, 3, 3)
    polarizability = zeros(T, 3, 3)
    δψ = Vector{typeof(scfres.ψ)}(undef, 3)
    for α = 1:3
        res = solve_ΩplusK_split(scfres.ham, scfres.ρ, scfres.ψ, scfres.occupation,
                                 scfres.εF, scfres.eigenvalues, ψ̄[α];
                                 scfres.occupation_threshold,
                                 bandtolalg=BandtolBalanced(scfres), q, kwargs...)
        δψ[α] = res.δψ
        for β = 1:3
            c = _dielectric_contraction(basis, ψ̄[β], res.δψ, scfres.occupation)
            ε∞[β, α] = (α == β) - (8T(π) / Ω) * c
            polarizability[β, α] = -2c  # Ω·χ_{βα}, i.e. the ε∞ term without 4π and 1/Ω
        end
    end

    (; ε∞, polarizability, δψ, ψ̄)
end

"""
Static per-cell polarizability tensor `Ω·χ` (`P = χ·E`) of an insulator; see
[`compute_dielectric`](@ref) for details and limitations.
"""
compute_polarizability(scfres::NamedTuple; kwargs...) =
    compute_dielectric(scfres; kwargs...).polarizability
