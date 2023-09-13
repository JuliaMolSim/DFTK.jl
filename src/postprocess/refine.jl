# Refinement of some quantities of interest (density, forces) following the
# strategy described in [CDKL2021].
# 
# [CDKL2021]:
#     E. Cancès, G. Dusson, G. Kemlin, and A. Levitt
#     *Practical error bounds for properties in plane-wave electronic structure
#     calculations* Preprint, 2021. [arXiv](https://arxiv.org/abs/2111.01470)

@kwdef struct PreRefinementOutputs
    basis_ref
    ψr
    ρr
    occupation
    schur_residual
    δρ
end

function refine_scfres(scfres, basis_ref::PlaneWaveBasis{T}; ΩpK_tol,
                       occ_threshold=default_occupation_threshold(T)) where {T}
    basis = scfres.basis

    @assert basis.model.lattice == basis_ref.model.lattice
    @assert length(basis.kpoints) == length(basis_ref.kpoints)
    @assert all(basis.kpoints[ik].coordinate == basis_ref.kpoints[ik].coordinate
                for ik in 1:length(basis.kpoints))

    haskey(scfres, :pre_refinement) && error() # TODO decide how to handle this...

    ψ, occ = select_occupied_orbitals(basis, scfres.ψ, scfres.occupation; threshold=occ_threshold)
    ψr = transfer_blochwave(ψ, basis, basis_ref)
    ρr = transfer_density(scfres.ρ, basis, basis_ref)
    _, hamr = energy_hamiltonian(basis_ref, ψr, occ; ρ=ρr)
    
    # Compute the residual R(P) and remove the virtual orbitals, as required
    # in src/scf/newton.jl
    
    # TODO fix compute_projected_gradient and replace
    res = [proj_tangent_kpt(hamr.blocks[ik] * ψk, ψk) for (ik, ψk) in enumerate(ψr)]

    # Compute M^{-1} R(P), with M^{-1} defined in [CDKL2021]
    P = [PreconditionerTPA(basis_ref, kpt) for kpt in basis_ref.kpoints]
    map(zip(P, ψr)) do (Pk, ψk)
        precondprep!(Pk, ψk)
    end

    function apply_M(φk, Pk, δφnk, n)
        proj_tangent_kpt!(δφnk, φk)
        δφnk = sqrt.(Pk.mean_kin[n] .+ Pk.kin) .* δφnk
        proj_tangent_kpt!(δφnk, φk)
        δφnk = sqrt.(Pk.mean_kin[n] .+ Pk.kin) .* δφnk
        proj_tangent_kpt!(δφnk, φk)
    end

    function apply_inv_M(φk, Pk, δφnk, n)
        proj_tangent_kpt!(δφnk, φk)
        op(x) = apply_M(φk, Pk, x, n)
        function f_ldiv!(x, y)
            x .= proj_tangent_kpt(y, φk)
            x ./= (Pk.mean_kin[n] .+ Pk.kin)
            proj_tangent_kpt!(x, φk)
        end
        J = LinearMap{eltype(φk)}(op, size(δφnk, 1))
        δφnk = IterativeSolvers.cg(J, δφnk, Pl=FunctionPreconditioner(f_ldiv!),
                  verbose=false, reltol=0, abstol=1e-15)
        proj_tangent_kpt!(δφnk, φk)
    end

    function apply_metric(φ, P, δφ, A::Function)
        map(enumerate(δφ)) do (ik, δφk)
            Aδφk = similar(δφk)
            φk = φ[ik]
            for n = 1:size(δφk,2)
                Aδφk[:,n] = A(φk, P[ik], δφk[:,n], n)
            end
            Aδφk
        end
    end

    # Compute the projection of the residual onto the high and low frequencies
    resLF = transfer_blochwave(res, basis_ref, basis)
    resHF = res - transfer_blochwave(resLF, basis, basis_ref)
    
    # - Compute M^{-1}_22 R_2(P)
    e2 = apply_metric(ψr, P, resHF, apply_inv_M)

    # Apply Ω+K to M^{-1}_22 R_2(P)
    Λ = map(enumerate(ψr)) do (ik, ψk)
        Hk = hamr.blocks[ik]
        Hψk = Hk * ψk
        ψk'Hψk
    end # Rayleigh coefficients
    ΩpKe2 = apply_Ω(e2, ψr, hamr, Λ) .+ apply_K(basis_ref, e2, ψr, ρr, occ)
    ΩpKe2 = transfer_blochwave(ΩpKe2, basis_ref, basis)
    
    # Invert Ω+K on the small space
    e1 = solve_ΩplusK(basis, ψ, resLF - ΩpKe2, occ; tol=ΩpK_tol).δψ
    e1 = transfer_blochwave(e1, basis, basis_ref)
    
    schur_residual = e1 + e2

    # Use the Schur residual to compute (minus) the first-order correction to
    # the density.
    δρ = compute_δρ(basis_ref, ψr, schur_residual, occ)

    merge(scfres, (pre_refinement = PreRefinementOutputs(; basis_ref, ψr, ρr, occupation=occ,
                                            schur_residual, δρ),))
end

function refine_density(scfres)
    haskey(scfres, :pre_refinement) || error() # TODO decide...
    scfres.pre_refinement.ρr - scfres.pre_refinement.δρ
end

function refine_forces(scfres; forces=nothing)
    haskey(scfres, :pre_refinement) || error() # TODO decide...
    isnothing(forces) && (forces = compute_forces(scfres)) # TODO use DiffResults?
    pre_ref = scfres.pre_refinement
    dF = ForwardDiff.derivative(ε -> compute_forces(pre_ref.basis_ref,
                                                    pre_ref.ψr.+ε.*pre_ref.schur_residual,
                                                    pre_ref.occupation;
                                                    ρ=pre_ref.ρr+ε.*pre_ref.δρ), 0)
    forces - dF
end
