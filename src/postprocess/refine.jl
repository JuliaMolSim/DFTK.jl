# Refinement of some quantities of interest (density, forces) following the
# strategy described in [CDKL2021].
#
# [CDKL2021]:
#     E. Cancès, G. Dusson, G. Kemlin, and A. Levitt
#     *Practical error bounds for properties in plane-wave electronic structure
#     calculations* Preprint, 2021. [arXiv](https://arxiv.org/abs/2111.01470)

"""
Result of calling the [`refine_scfres`](@ref) function.
- `basis`: Refinement basis, larger than the basis used to
           run a first [`self_consistent_field`](@ref) computation.
- `ψ`, `ρ`, `occupation`: Quantities from the scfres, transferred to the refinement basis
                          and with virtual orbitals removed.
- `δψ`, `δρ`: First order corrections to the wavefunctions and density.
              The sign is such that the refined quantities are ψ - δψ and ρ - δρ.
"""
struct RefinementResult
    basis::PlaneWaveBasis
    ψ
    ρ
    occupation
    δψ
    δρ
end

"""
Transfer the result of an SCF to a larger basis set,
and compute first order corrections ("refinements") to the wavefunctions and density.

Only full occupations are currently supported.

Returns a [`RefinementResult`](@ref) instance that can be used to refine quantities of interest,
through [`refine_density`](@ref) and [`refine_forces`](@ref).
"""
function refine_scfres(scfres, basis_ref::PlaneWaveBasis{T}; ΩpK_tol,
                       occ_threshold=default_occupation_threshold(T), kwargs...) where {T}
    basis = scfres.basis

    @assert basis.model.lattice == basis_ref.model.lattice
    @assert length(basis.kpoints) == length(basis_ref.kpoints)
    @assert all(basis.kpoints[ik].coordinate == basis_ref.kpoints[ik].coordinate
                for ik in 1:length(basis.kpoints))

    # Virtual orbitals must be removed
    ψ, occ = select_occupied_orbitals(basis, scfres.ψ, scfres.occupation; threshold=occ_threshold)
    check_full_occupation(basis, occ)

    ψr = transfer_blochwave(ψ, basis, basis_ref)
    ρr = transfer_density(scfres.ρ, basis, basis_ref)
    _, hamr = energy_hamiltonian(basis_ref, ψr, occ; ρ=ρr)

    # Compute the residual R(P)
    # This recomputes ρr internally, but it's only a minor inefficiency.
    res = compute_projected_gradient(basis_ref, ψr, occ)

    # Compute the projection of the residual onto the high and low frequencies
    resLF = transfer_blochwave(res, basis_ref, basis)
    resHF = res - transfer_blochwave(resLF, basis, basis_ref)

    """
    Apply M^{-1}.

    M applied to the i-th band is defined by
      M_i = P^⟂ T_i^{1/2} P^⟂ T_i^{1/2} P^⟂,
    with the diagonal operator
      T_i = kinetic energy + mean kinetic energy of i-th band.

    To invert M_i, use P^⟂ T_i^{-1} P^⟂ as the preconditioner.

    See [CDKL2021] for more details.
    """
    function apply_inv_M(ψ, res)
        """Apply the M_i operator with i=n."""
        function apply_M(ψk, Pk, δψnk, n)
            proj_tangent_kpt!(δψnk, ψk)
            δψnk = sqrt.(Pk.mean_kin[n] .+ Pk.kin) .* δψnk
            proj_tangent_kpt!(δψnk, ψk)
            δψnk = sqrt.(Pk.mean_kin[n] .+ Pk.kin) .* δψnk
            proj_tangent_kpt!(δψnk, ψk)
        end

        """Apply the M_i^{-1} operator with i=n."""
        function apply_inv_M(ψk, Pk, resk, n)
            proj_tangent_kpt!(resk, ψk)
            op(x) = apply_M(ψk, Pk, x, n)
            function f_ldiv!(x, y)
                x .= proj_tangent_kpt(y, ψk)
                x ./= (Pk.mean_kin[n] .+ Pk.kin)
                proj_tangent_kpt!(x, ψk)
            end
            J = LinearMap{eltype(ψk)}(op, size(resk, 1))
            δψk = IterativeSolvers.cg(J, resk, Pl=FunctionPreconditioner(f_ldiv!),
                      verbose=false, reltol=0, abstol=1e-15)
            proj_tangent_kpt!(δψk, ψk)
        end

        map(enumerate(res)) do (ik, resk)
            P = PreconditionerTPA(basis_ref, basis_ref.kpoints[ik])
            ψk = ψ[ik]
            precondprep!(P, ψk)
            δψk = similar(resk)
            for n = 1:size(resk, 2)
                # Apply M_i^{-1} to each band.
                δψk[:, n] = apply_inv_M(ψk, P, resk[:, n], n)
            end
            δψk
        end
    end

    # Compute M^{-1}_22 R_2(P) as an approximation of (Ω+K)^{-1}_22 R_2(P).
    e2 = apply_inv_M(ψr, resHF)

    # Apply Ω+K to M^{-1}_22 R_2(P)
    Λ = map(enumerate(ψr)) do (ik, ψk)
        Hk = hamr.blocks[ik]
        Hψk = Hk * ψk
        ψk'Hψk
    end # Rayleigh coefficients
    ΩpKe2 = apply_Ω(e2, ψr, hamr, Λ) .+ apply_K(basis_ref, e2, ψr, ρr, occ)
    ΩpKe2 = transfer_blochwave(ΩpKe2, basis_ref, basis)

    rhs = resLF - ΩpKe2

    # Invert Ω+K on the small space
    e1 = solve_ΩplusK(basis, ψ, rhs, occ; tol=ΩpK_tol).δψ

    e1 = transfer_blochwave(e1, basis, basis_ref)
    schur_residual = e1 + e2

    # Use the Schur residual to compute (minus) the first-order correction to
    # the density.
    δρ = compute_δρ(basis_ref, ψr, schur_residual, occ)

    RefinementResult(basis_ref, ψr, ρr, occ, schur_residual, δρ)
end

"""
Retrieve the refined density from a [`RefinementResult`](@ref).
"""
function refine_density(refinement::RefinementResult)
    refinement.ρ - refinement.δρ
end

"""
Refine forces using a [`RefinementResult`](@ref).

Either the unrefined forces must be provided, or an `scfres` to compute them.
"""
function refine_forces(refinement::RefinementResult, forces::AbstractArray)
    dF = ForwardDiff.derivative(ε -> compute_forces(refinement.basis,
                                                    refinement.ψ .+ ε.*refinement.δψ,
                                                    refinement.occupation;
                                                    ρ=refinement.ρ + ε.*refinement.δρ), 0)
    forces - dF
end
function refine_forces(refinement::RefinementResult, scfres::NamedTuple)
    refine_forces(refinement, compute_forces(scfres)) # TODO use DiffResults?
end
