using LinearMaps
using IterativeSolvers

# Newton's algorithm to solve SCF equations
#
# Newton algorithm consists of iterating over density matrices like
#       P <- P + δP, with additional renormalization
# where δP solves ([1])
#       (Ω+K)δP = -[P, [P, H(P)]]
# where (Ω+K) is the constrained Hessian of the energy. It is
# defined as a super operator on the space tangent to the constraints
# manifold at point P_∞, the solution of our problem.
#   - K is the unconstrained Hessian of the energy on the manifold;
#   - Ω represents the influence of the curvature of the manifold :
#         ΩδP = -[P_∞, [H(P_∞), δP]].
#     In practice, we dont have access to P_∞ so we just use the current P.
#     Another way to see Ω is to link it to the four-point independent-particle
#     susceptibility. Indeed, if we define the following extension and
#     retraction operators
#       - E : builds a density matrix Eρ given a density ρ via Eρ(r,r') = δ(r,r')ρ(r),
#       - R : builds a density ρ given a density matrix O via ρ(r) = O(r,r),
#     then we have the relation χ0₂ = Rχ0₄E, where χ0₂ is the two-point
#     independent-particle susceptibility (it returns δρ from a given δV) and
#     χ0₄ = -Ω^{-1} is the four-point independent-particle susceptibility.
#
# For further details :
# [1] Eric Cancès, Gaspard Kemlin, Antoine Levitt. Convergence analysis of
#     direct minimization and self-consistent iterations. SIAM Journal of Matrix
#     Analysis and Applications, 42(1):243–274 (2021)
#
# We presented the algorithm in the density matrix framework, but in practice,
# we implement it with orbitals.
# In this framework, an element on the tangent space at ψ can be written as
#       δP = Σ |ψi><δψi| + hc
# where the δψi are of size Nb and are all orthogonal to the ψj, 1 <= j <= N.
# Therefore we store them in the same kind of array than ψ, with
# δψ[ik][:, i] = δψi for each k-point.
# In this framework :
#   - computing Ωδψ can be done analytically with the formula
#         Ωδψi = H*δψi - Σj λji δψj
#     where λij = <ψi|H|ψj>. This is one way among others to extend the
#     definition of ΩδP = -[P_∞, [H(P_∞), δP]] to any point P on the manifold --
#     we chose this one because it makes Ω self-adjoint for the Frobenius scalar
#     product;
#   - computing Kδψ can be done by constructing the δρ associated with δψ, then
#     using the exchange-correlation kernel to compute the associated δV, then
#     acting with δV on the ψ and projecting on the tangent space.

#  Compute the gradient of the energy, projected on the space tangent to ψ, that
#  is to say H(ψ)*ψ - ψ*λ where λ is the set of Rayleigh coefficients associated
#  to the ψ.
function compute_projected_gradient(basis::PlaneWaveBasis, ψ, occupation)
    ρ = compute_density(basis, ψ, occupation)
    _, H = energy_hamiltonian(basis, ψ, occupation; ρ=ρ)

    [proj_tangent_kpt(H.blocks[ik] * ψk, ψk) for (ik, ψk) in enumerate(ψ)]
end

# Projections on the space tangent to ψ
function proj_tangent_kpt!(δψk, ψk)
    # δψk = δψk - ψk * (ψk'δψk)
    mul!(δψk, ψk, ψk'δψk, -1, 1)
end
proj_tangent_kpt(δψk, ψk) = proj_tangent_kpt!(deepcopy(δψk), ψk)

function proj_tangent(δψ, ψ)
    [proj_tangent_kpt(δψ[ik], ψk) for (ik, ψk) in enumerate(ψ)]
end
function proj_tangent!(δψ, ψ)
    [proj_tangent_kpt!(δψ[ik], ψk) for (ik, ψk) in enumerate(ψ)]
end


"""
    newton(basis::PlaneWaveBasis{T}; ψ0=nothing,
           tol=1e-6, tol_cg=1e-10, maxiter=20, verbose=false,
           callback=NewtonDefaultCallback(),
           is_converged=NewtonConvergenceDensity(tol))

Newton algorithm. Be careful that the starting point needs to be not too far
from the solution.
"""
function newton(basis::PlaneWaveBasis{T}, ψ0;
                tol=1e-6, tol_cg=1e-10, maxiter=20, verbose=false,
                callback=ScfDefaultCallback(),
                is_converged=ScfConvergenceDensity(tol)) where T

    # setting parameters
    model = basis.model
    @assert model.temperature == 0 # temperature is not yet supported

    # check that there are no virtual orbitals
    filled_occ = filled_occupation(model)
    n_spin = model.n_spin_components
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)
    @assert n_bands == size(ψ0[1], 2)

    # number of kpoints and occupation
    Nk = length(basis.kpoints)
    occupation = [filled_occ * ones(T, n_bands) for ik = 1:Nk]

    # iterators
    n_iter = 0

    # orbitals, densities and energies to be updated along the iterations
    ψ = deepcopy(ψ0)
    ρ = compute_density(basis, ψ, occupation)
    energies, H = energy_hamiltonian(basis, ψ, occupation; ρ=ρ)
    converged = false

    # perform iterations
    while !converged && n_iter < maxiter
        n_iter += 1

        # compute Newton step and next iteration
        res = compute_projected_gradient(basis, ψ, occupation)
        # solve (Ω+K) δψ = -res so that the Newton step is ψ <- ψ + δψ
        δψ = solve_ΩplusK(basis, ψ, -res, occupation; tol_cg, verbose).δψ
        ψ  = [ortho_qr(ψ[ik] + δψ[ik]) for ik in 1:Nk]

        ρ_next = compute_density(basis, ψ, occupation)
        energies, H = energy_hamiltonian(basis, ψ, occupation; ρ=ρ_next)
        info = (ham=H, basis=basis, converged=converged, stage=:iterate,
                ρin=ρ, ρout=ρ_next, n_iter=n_iter, energies=energies, algorithm="Newton")
        callback(info)

        # update and test convergence
        converged = is_converged(info)
        ρ = ρ_next
    end

    # Rayleigh-Ritz
    eigenvalues = []
    for ik = 1:Nk
        Hψk = H.blocks[ik] * ψ[ik]
        F = eigen(Hermitian(ψ[ik]'Hψk))
        push!(eigenvalues, F.values)
        ψ[ik] .= ψ[ik] * F.vectors
    end

    εF = nothing  # does not necessarily make sense here, as the
                  # Aufbau property might not even be true

    # return results and call callback one last time with final state for clean
    # up
    info = (ham=H, basis=basis, energies=energies, converged=converged,
            ρ=ρ, eigenvalues=eigenvalues, occupation=occupation, εF=εF,
            n_iter=n_iter, ψ=ψ, stage=:finalize, algorithm="Newton")
    callback(info)
    info
end
