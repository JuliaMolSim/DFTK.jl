@doc raw"""
Uniform electric field. Adds the following total energy contribution,
thereby turning the total energy into the electric enthalpy:
```math
-\Omega \bm{\mathcal E} \cdot \mathbf{P}_\mathrm{tot},
```
where $\Omega$ is the unit cell volume, $\bm{\mathcal E}$ is the
electric field strength, and $\mathbf{P}_\mathrm{tot}$ is the total
(electronic + ionic) polarization.
"""
struct ElectricField
    strength_red
    polarization_method::PolarizationMethod

    function ElectricField(strength_red, polarization_method::PolarizationMethod)
        if any(ForwardDiff.value.(strength_red) .!= 0)
            error("ElectricField term only supports infinitesimal fields,"
                  * " but received finite field strength: $strength_red.")
        end
        new(strength_red, polarization_method)
    end
end

function (e::ElectricField)(basis::PlaneWaveBasis)
    TermElectricField(e.strength_red, e.polarization_method)

    # TODO: this should check the model's symmetries and make sure that none of them is broken
end

struct TermElectricField{TS,TPol}
    strength_red::TS
    polarization_method::TPol
end

@timing "ene_ops: electric field" function ene_ops(term::TermElectricField,
                                                   basis::PlaneWaveBasis{T},
                                                   ψ, occupation;
                                                   ρ=nothing, τ=nothing) where {T}
    # For DFPT purposes, always returning zero energy is fine for now
    E = zero(T)
    if iszero(term.strength_red)
        return (; E, ops=[NoopOperator(basis, kpt) for kpt in basis.kpoints])
    end

    # Let w = i * ∑_α Eα * du/dkα
    # We need w = (w u') u
    # we have <w, u> = 0 from how the Sternheimer equation is solved, so
    # let's use the following self-adjoint operator:
    # wu' + u'w
    # [w u] [0 1; 1 0] [w u]'
    lrpol = term.polarization_method::LinearResponsePolarization

    # TODO: Recursive Hamiltonian construction -> wtf? :D
    # TODO: not available at Hamiltonian construction time (for a good reason)
    occupation_threshold = default_occupation_threshold(T)
    ham = Hamiltonian(basis; ψ, occupation, ρ, τ)
    bandtolalg = BandtolBalanced(basis, ψ, occupation; occupation_threshold)

    lrpol.verbose && mpi_master(basis0.comm_kpts) && println("Computing dunk/dk_$α")

    dψ_dk = map(1:3) do α
        δk = zeros(V, 3)
        δk[α] = one(V)
        dH_dkψ = compute_δHψδk(basis0, ψ0, δk; ρ=ρ0, τ=τ0)
        apply_χ0_4P(ham, ψ0, occupation0, εF0, eigenvalues0, dH_dkψ;
                    occupation_threshold, bandtolalg, lrpol.tol).δψ
    end
end

function compute_forces(term::TermElectricField, basis::PlaneWaveBasis{T}, ψ, occupation; kwargs...)
    map(basis.model.atoms) do atom
        charge_ionic(atom) * 1/2π * Vec3{T}(term.strength_red)
    end
end