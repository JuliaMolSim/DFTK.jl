include("../postprocess/polarization.jl")

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

# TODO: is this a nonlinear term?
struct TermElectricField{TS,TPol} <: Term
    strength_red::TS
    polarization_method::TPol
end

# Primal field strength: the constructor guarantees that the field is exactly zero.
function ene_ops(term::TermElectricField, basis::PlaneWaveBasis{T}, ψ, occupation;
                 kwargs...) where {T}
    (; E=zero(T), ops=[NoopOperator(basis, kpt) for kpt in basis.kpoints])
end

# Infinitesimal (dual) field strength: return an operator whose dual part is the
# DFPT right-hand side for the electric field perturbation.
#
# For each partial n, let δE⁽ⁿ⁾ be the partials of the reduced field strength.
# The derivative of the enthalpy -E ⋅ P wrt. <uₙₖ| (divided by wₖfₙₖ) is
#   wₙₖ = i/2π ∑_α δE⁽ⁿ⁾_α ∂k_α uₙₖ = i/2π ∂_{δE⁽ⁿ⁾} uₙₖ,
# i.e. a single k-derivative in direction δk = δE⁽ⁿ⁾. We represent this as the
# self-adjoint operator ∑ₙ (|wₙ><uₙ| + |uₙ><wₙ|) over the occupied bands, which satisfies
# V uₙ = wₙ since <wₙ|uₘ> = 0 for occupied bands (parallel-transport gauge).
# The ψ passed in may be primal or dual: since the field value is zero, only the primal
# part of ψ contributes to first order.
#
# The energy is not computed (always zero), only the operator is meaningful.
@timing "ene_ops: electric field" function ene_ops(term::TermElectricField,
                                                   basis::PlaneWaveBasis{T},
                                                   ψ, occupation;
                                                   ρ=nothing, τ=nothing,
                                                   eigenvalues=nothing, εF=nothing,
                                                   kwargs...) where {Tg,V,N,T<:Dual{Tg,V,N}}
    E = zero(T)
    # The strength may also be primal (e.g. when differentiating wrt. other parameters)
    δE = map(1:N) do n
        Vec3{V}(map(Eα -> Eα isa Dual ? ForwardDiff.partials(Tg, Eα, n) : zero(V),
                    term.strength_red))
    end
    if isnothing(ψ) || all(iszero, δE)
        return (; E, ops=[NoopOperator(basis, kpt) for kpt in basis.kpoints])
    end
    if isnothing(eigenvalues) || isnothing(εF)
        error("ElectricField term requires the eigenvalues and εF to build the Hamiltonian.")
    end
    if length(basis.model.symmetries) > 1
        error("ElectricField term does not support symmetries yet, use symmetries=false.")
    end
    lrpol = term.polarization_method::LinearResponsePolarization

    (; basis0, ψ0, occupation0, εF0, eigenvalues0, ρ0, τ0) =
        _primal_scf_quantities(Tg, basis, ψ, occupation, εF, eigenvalues, ρ, τ)
    occupation_threshold = default_occupation_threshold(V)
    ham = Hamiltonian(basis0; ψ=ψ0, occupation=occupation0, ρ=ρ0, τ=τ0)
    bandtolalg = BandtolBalanced(basis0, ψ0, occupation0; occupation_threshold)

    # TODO: The ForwardDiff SCF rule calls ene_ops twice with the same primal ψ,
    #       so these Sternheimer solves are done twice. Could be cached.
    dψ_dδE = map(1:N) do n
        lrpol.verbose && mpi_master(basis0.comm_kpts) &&
            println("Computing dunk/dk in direction $(δE[n])")
        compute_dψdk(basis0, ψ0, occupation0, εF0, eigenvalues0, δE[n]; ρ=ρ0, τ=τ0,
                     occupation_threshold, lrpol.tol, ham, bandtolalg)
    end

    ops = map(1:length(basis.kpoints)) do ik
        occ_idx = findall(>(occupation_threshold), occupation0[ik])
        u = ψ0[ik][:, occ_idx]
        w_partials = [im / 2V(π) * dψ_dδE[n][ik][:, occ_idx] for n in 1:N]
        w = map(w_partials...) do wn...
            Complex(Dual{Tg}(zero(V), real.(wn)), Dual{Tg}(zero(V), imag.(wn)))
        end
        n_occ = length(occ_idx)
        # P D P' = |w><u| + |u><w|
        P = hcat(w, u)
        D = [zeros(V, n_occ, n_occ) I; I zeros(V, n_occ, n_occ)]
        NonlocalOperator(basis, basis.kpoints[ik], P, D)
    end
    (; E, ops)
end

function energy(term::TermElectricField, basis::PlaneWaveBasis{T}, ψ, occupation;
                kwargs...) where {T}
    zero(T)
end

function compute_forces(term::TermElectricField, basis::PlaneWaveBasis{T}, ψ, occupation; kwargs...) where {T}
    map(basis.model.atoms) do atom
        charge_ionic(atom) * Vec3{T}(term.strength_red)
    end
end

function apply_kernel(term::TermElectricField, basis::PlaneWaveBasis, δρ; kwargs...)
    iszero(term.strength_red) || error("Cannot apply kernel at nonzero field: $(term.strength_red)")
    (;)
end