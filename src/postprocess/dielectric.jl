# Homogeneous electric-field response of an insulator: the electronic (clamped-ion)
# dielectric tensor ε∞ and the static polarizability, from density-functional perturbation
# theory.
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
"""
function compute_δHψ_dk(basis::PlaneWaveBasis{T}, ψ, ρ, τ) where {T}
    # A unit Cartesian displacement of k is the reduced shift recip_lattice⁻¹ eα, since
    # k_red = recip_lattice⁻¹ k_cart; differentiating H along it gives ∂H/∂k_cart,α.
    Binv = inv(basis.model.recip_lattice)
    map(1:3) do α
        # Shift all k-points by t·(reduced eα) and apply the shifted Hamiltonian to the fixed
        # orbitals; t enters only through the operators' explicit k-dependence (kinetic, nonlocal,
        # and for meta-GGA the DivAgrad term — hence the frozen τ). Differentiating at t=0 gives
        # (∂H_k/∂k_α) ψ per k-point, and [H_k, r_α] ψ = i (∂H_k/∂k_α) ψ.
        dHψ = ForwardDiff.derivative(zero(T)) do t
            ham_t = Hamiltonian(shift_kpoints(basis, t .* Vec3(Binv[:, α])); ρ, τ)
            [ham_t[ik] * ψ[ik] for ik = 1:length(basis.kpoints)]
        end
        im .* dHψ
    end
end

function _assert_insulator(scfres)
    filled = filled_occupation(scfres.basis.model)
    for occ_k in scfres.occupation, occ in occ_k
        empty = occ < scfres.occupation_threshold
        full  = occ > filled - scfres.occupation_threshold
        empty || full || error(
            "Electric-field response is implemented for insulators only.")
    end
end

@doc raw"""
Compute the electronic (clamped-ion) **dielectric tensor** ``ε∞`` and the static
**polarizability** of an insulator from its response to a homogeneous electric field
(density-functional perturbation theory). Returns a NamedTuple
`(; ε∞, polarizability, ψ̄)` where
```math
ε∞[β,α] = δ_{βα} - \frac{8π}{Ω} ∑_k w_k ∑_n f_n \operatorname{Re}⟨ψ̄^β_{nk} | δψ^α_{nk}⟩,
```
`polarizability` `= Ω·χ` is the per-cell polarizability (``P = χ·E``) and `ψ̄[α]` are the
projected-position orbitals ``P_c r_α ψ``.
"""
@timing function compute_dielectric(scfres::NamedTuple; kwargs...)
    basis = scfres.basis
    T = eltype(basis)
    _assert_insulator(scfres)
    length(basis.symmetries) == 1 || error(
        "compute_dielectric requires a symmetry-free basis (for now); build the model with " *
        "`Model(system; symmetries=false)`.")

    Ω = basis.model.unit_cell_volume
    q = zero(Vec3{T})

    # ψ̄^α = P_c r_α ψ, the position operator applied to the occupied orbitals and projected
    # onto the unoccupied space. Under periodic boundary conditions r_α ψ is ill-defined, so we
    # obtain ψ̄^α from the well-defined velocity [H, r_α]ψ = i ∂H/∂k_α ψ by one non-interacting
    # Sternheimer solve (apply_χ0_4P inverts H - εₙ on the conduction space, i.e. supplies the
    # 1/(εₘ - εₙ) energy denominators that turn the velocity into the position). This ψ̄^α is
    # exactly the perturbation-times-orbitals that the self-consistent solver expects as input:
    # we cannot hand solve_ΩplusK_split the velocity δHψ directly, as that is the commutator
    # [H, r_α]ψ, not r_α ψ.
    δHψ = compute_δHψ_dk(basis, scfres.ψ, scfres.ρ, get(scfres, :τ, nothing))
    ψ̄ = map(1:3) do α
        apply_χ0_4P(scfres.ham, scfres.ψ, scfres.occupation, scfres.εF, scfres.eigenvalues,
                    δHψ[α]; q, scfres.occupation_threshold,
                    bandtolalg=BandtolBalanced(scfres)).δψ
    end

    # δψ^α: full self-consistent (Ω+K) response to the field along α, then contract against ψ̄^β.
    ε∞ = zeros(T, 3, 3)
    polarizability = zeros(T, 3, 3)
    for α = 1:3
        δψ = solve_ΩplusK_split(scfres.ham, scfres.ρ, scfres.ψ, scfres.occupation,
                                scfres.εF, scfres.eigenvalues, ψ̄[α];
                                scfres.occupation_threshold,
                                bandtolalg=BandtolBalanced(scfres), q, kwargs...).δψ
        for β = 1:3
            # c = ∑_k w_k ∑_n f_n Re⟨ψ̄^β_nk | δψ^α_nk⟩ (spin channels are summed by weighted_ksum)
            per_k = map(1:length(basis.kpoints)) do ik
                sum(scfres.occupation[ik][n] * real(dot(ψ̄[β][ik][:, n], δψ[ik][:, n]))
                    for n = 1:size(δψ[ik], 2); init=zero(T))
            end
            c = weighted_ksum(basis, per_k)
            ε∞[β, α] = (α == β) - (8T(π) / Ω) * c
            polarizability[β, α] = -2c  # Ω·χ_{βα}, i.e. the ε∞ term without 4π and 1/Ω
        end
    end

    (; ε∞, polarizability, ψ̄)
end
