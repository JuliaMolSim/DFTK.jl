# Refinement of some quantities of interest (density, forces) following the
# strategy described in [CDKL2022].
# We use a different sign convention than [CDKL2022] for corrections δX,
# such that X + δX is the refined quantity.
#
# The overall strategy is to perform a first SCF,
# transfer the result to a new basis set with a larger Ecut,
# then perform an approximate first order correction in the larger space.
#
# To compute the first order correction ("refinement") δP in the larger space,
# one needs to invert (Ω+K)δP = -R(P) where (Ω+K) is the Hessian matrix and
# R(P) = [P, [P, H(P)]] is the residual.
# See newton.jl for an explanation of these quantities.
#
# The basis elements of the smaller basis set are called low frequency (_1 suffix).
# The additional basis elements are all high frequency (_2 suffix).
# We approximate (Ω+K)_22 by M_22 which is cheap to compute and invert, and (Ω+K)_21 by 0.
# The motivation is that the dominant term in Ω+K for high frequencies is the kinetic energy.
#
# The metric operator M is as follows:
# M applied to the n-th band is defined by
#   M_n = P^⟂ T_n^{1/2} P^⟂ T_n^{1/2} P^⟂,
# with the diagonal operator
#   T_n = kinetic energy + mean kinetic energy of n-th band.
# To invert M_n effectively, we use P^⟂ T_n^{-1} P^⟂ as the preconditioner.
#
# Solving the system then amounts to computing
#   δP_2 = -M^{-1}_22 R_2(P)
#   δP_1 = (Ω+K)^{-1}_11 (-R_1(P) - (Ω+K)_12 δP_2)
# The inversion of (Ω+K)_11 is roughly as expensive as the SCF on the low frequency space.
#
# [CDKL2022]:
#     E. Cancès, G. Dusson, G. Kemlin, and A. Levitt
#     *Practical error bounds for properties in plane-wave electronic structure calculations*
#     [SIAM Journal on Scientific Computing 44 (5), B1312-B1340](https://doi.org/10.1137/21M1456224)

import DifferentiationInterface: AutoForwardDiff, value_and_derivative

"""
Invert the metric operator M.
"""
@timing function invert_refinement_metric(basis::PlaneWaveBasis{T}, ψ, res) where {T}
    # Apply the M_n operator.
    function apply_M!(ψk, Pk, δψnk, n)
        proj_tangent_kpt!(δψnk, ψk)
        δψnk = sqrt.(Pk.mean_kin[n] .+ Pk.kin) .* δψnk
        proj_tangent_kpt!(δψnk, ψk)
        δψnk = sqrt.(Pk.mean_kin[n] .+ Pk.kin) .* δψnk
        proj_tangent_kpt!(δψnk, ψk)
    end

    # Apply the M_n^{-1} operator.
    function apply_inv_M(ψk, Pk, resk, n)
        resk = proj_tangent_kpt(resk, ψk)
        function op!(Mx, x)
            Mx .= apply_M!(ψk, Pk, x, n)
        end
        function f_ldiv!(x, y)
            x .= proj_tangent_kpt(y, ψk)
            x ./= (Pk.mean_kin[n] .+ Pk.kin)
            proj_tangent_kpt!(x, ψk)
        end
        # This CG seems to converge very quickly in practice (often 1 iteration).
        cginfo = cg(op!, resk;
                    precon=FunctionPreconditioner(f_ldiv!),
                    tol=100*eps(T), maxiter=20)
        if !cginfo.converged
            @warn """CG for `invert_refinement_metric` did not converge, this is unexpected.
                    Residual norm after $(cginfo.n_iter) iterations is $(cginfo.residual_norm)."""
        end
        proj_tangent_kpt!(cginfo.x, ψk)
    end

    map(basis.kpoints, ψ, res) do kpt, ψk, resk
        P = PreconditionerTPA(basis, kpt)
        precondprep!(P, ψk)
        δψk = similar(resk)
        for n = 1:size(resk, 2)
            # Apply M_n^{-1} to each band.
            δψk[:, n] = @views apply_inv_M(ψk, P, resk[:, n], n)
        end
        δψk
    end
end

"""
Result of calling the [`refine_scfres`](@ref) function.

- `basis`: Refinement basis, larger than the basis used to
           run a first [`self_consistent_field`](@ref) computation.
- `ψ`, `ρ`, `occupation`: Quantities from the scfres, transferred to the refinement basis
                          and with virtual orbitals removed.
- `δψ`, `δρ`: First order corrections to the wavefunctions and density.
              The refined quantities are ψ + δψ and ρ + δρ.
- `ΩpK_res`: Additional information returned by the inversion of (Ω+K)_11.
"""
struct RefinementResult{T}
    basis::PlaneWaveBasis{T}
    ψ
    ρ
    occupation
    δψ
    δρ
    ΩpK_res
end

"""
Transfer the result of an SCF to a larger basis set,
and compute approximate first order corrections ("refinements") to the wavefunctions and density.

Only full occupations are currently supported.

Returns a [`RefinementResult`](@ref) instance that can be used to refine quantities of interest,
through [`refine_energies`](@ref) and [`refine_forces`](@ref).
"""
@timing function refine_scfres(scfres, basis_ref::PlaneWaveBasis{T};
                               tol=1e-6,
                               occ_threshold=default_occupation_threshold(T),
                               kwargs...) where {T}
    basis = scfres.basis

    @assert basis.model.lattice == basis_ref.model.lattice
    @assert length(basis.kpoints) == length(basis_ref.kpoints)
    @assert all(basis.kpoints[ik].coordinate == basis_ref.kpoints[ik].coordinate
                for ik = 1:length(basis.kpoints))

    # Virtual orbitals must be removed
    ψ, occ = select_occupied_orbitals(basis, scfres.ψ, scfres.occupation; threshold=occ_threshold)
    check_full_occupation(basis, occ)

    ψr = transfer_blochwave(ψ, basis, basis_ref)
    ρr = transfer_density(scfres.ρ, basis, basis_ref)
    _, hamr = energy_hamiltonian(basis_ref, ψr, occ; ρ=ρr)

    # Compute (minus) the residual -R(P)
    # This recomputes ρr internally, but it's only a minor inefficiency.
    res = -compute_projected_gradient(basis_ref, ψr, occ)

    # Compute the projection of the residual onto the high and low frequencies
    resLF = transfer_blochwave(res, basis_ref, basis)
    resHF = res - transfer_blochwave(resLF, basis, basis_ref)

    # Compute -M^{-1}_22 R_2(P) as an approximation of -(Ω+K)^{-1}_22 R_2(P).
    e2 = invert_refinement_metric(basis_ref, ψr, resHF)

    # Apply Ω+K to -M^{-1}_22 R_2(P)
    @timing "Λ computation" Λ = map(hamr.blocks, ψr) do Hk, ψk
        Hψk = Hk * ψk
        ψk'Hψk
    end # Rayleigh coefficients
    ΩpKe2 = apply_Ω(e2, ψr, hamr, Λ) .+ apply_K(basis_ref, e2, ψr, ρr, occ)
    ΩpKe2 = transfer_blochwave(ΩpKe2, basis_ref, basis)

    rhs = ΩpKe2 - resLF

    # Invert Ω+K on the small space
    ΩpK_res = solve_ΩplusK(basis, ψ, rhs, occ; tol, kwargs...)

    e1 = transfer_blochwave(ΩpK_res.δψ, basis, basis_ref)
    schur_residual = e1 + e2

    # Use the Schur residual to compute the first-order correction to the density.
    δρ = compute_δρ(basis_ref, ψr, schur_residual, occ)

    ΩpK_res = Base.structdiff(ΩpK_res, NamedTuple{(:δψ,)}) # remove δψ from res tuple
    RefinementResult(basis_ref, ψr, ρr, occ, schur_residual, δρ, ΩpK_res)
end

"""
Refine energies using a [`RefinementResult`](@ref).

The refined energies can be obtained by E + dE.
"""
@timing function refine_energies(refinement::RefinementResult{T}) where {T}
    term_names = [string(nameof(typeof(term))) for term in refinement.basis.model.term_types]

    f(ε) = energy(refinement.basis,
                  refinement.ψ + ε.*refinement.δψ,
                  refinement.occupation;
                  ρ=refinement.ρ + ε.*refinement.δρ).energies.values
    E, dE = value_and_derivative(f, AutoForwardDiff(), zero(T))
    (; E=Energies(term_names, E), dE=Energies(term_names, dE))
end

"""
Refine forces using a [`RefinementResult`](@ref).

The refined forces can be obtained by F + dF.
"""
@timing function refine_forces(refinement::RefinementResult{T}) where {T}
    # Arrays of arrays are not officially supported by ForwardDiff.
    # Reinterpret the Vector{SVector{3}} as a flat vector for differentiation.
    pack(x) = reinterpret(eltype(eltype(x)), x) # eltype is a Dual not just T!
    unpack(x) = reinterpret(SVector{3, T}, x)

    f(ε) = pack(compute_forces(refinement.basis,
                               refinement.ψ .+ ε.*refinement.δψ,
                               refinement.occupation;
                               ρ=refinement.ρ + ε.*refinement.δρ))
    F, dF = value_and_derivative(f, AutoForwardDiff(), zero(T))

    (; F=unpack(F), dF=unpack(dF))
end
