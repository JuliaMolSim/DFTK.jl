# Newton's algorithm for the direct minimization of the energy
# ------------------------------------------------------------
#
# Newton algorithm consist of iterating over density matrices like
#       P = P - δP
# where δP solves
#       (Ω+K)δP = [P,[P,H(P)]] (residual)
# where (Ω+K) if the Jacobian of the direct minimization iterations. It is
# defined as a super operator from the space tangent to the constraints
# manifold at point P_∞, the solution of our problem.
#   - K is the Hessian of the energy on the manifold;
#   - Ω represents the influence of the curvature of the manifold :
#         ΩδP = -[P_∞,[H(P_∞),δP]].
#     In practice, we dont have access to P_∞ so we just use the current P.
#     Ω is also -χ0^(-1) where χ0 is the occupation function.
#
# Here, in the orbital framework, an element on the tangent space at φ can be written as
#       δP = Σ |φi><δφi| + hc
# where the δφi are of size Nb and are all orthogonal to the φj, 1 <= j <= N.
# Therefore we store them in the same kind of array than φ, with
# δφ[ik][:,i] = δφi for each k-point.
# In this framework :
#   - computing Ωδφ can be done analitically with the formula
#         Ωδφi = H*δφi - Σj <δφj|H|δφi> δφj - Σj λji δφj
#     where λij = <φi|H|φj>;
#   - computing Kδφ can be done using the kernel function of DFTK after
#     computing the associated δρ;
#   - to perform Newton iterations, we solve (Ω+K)δφ = Hφ-λφ and then perform
#     φ = φ - δφ, with proper orthonormalization.
#
# For further details :
# Eric Cancès, Gaspard Kemlin, Antoine Levitt. Convergence analysis of
# direct minimization and self-consistent iterations. SIAM Journal of Matrix
# Analysis and Applications, 42(1):243–274 (2021)

import KrylovKit: ArnoldiIterator, Orthogonalizer, OrthonormalBasis, KrylovDefaults, orthogonalize!
using KrylovKit

#
# Tools
#

"""
    compute_residual(basis::PlaneWaveBasis, φ, occ)

Compute the residual associated to a set of planewave φ, that is to say
H(φ)*φ - λ*φ where λ is the set of rayleigh coefficients associated to the φ.
"""
function compute_residual(basis::PlaneWaveBasis, φ, occ)

    # necessary quantities
    Nk = length(basis.kpoints)
    ρ = compute_density(basis, φ, occ)
    energies, H = energy_hamiltonian(basis, φ, occ; ρ=ρ)

    # compute residual
    res = similar(φ)
    for ik = 1:Nk
        φk = φ[ik]
        N = size(φk, 2)
        Hk = H.blocks[ik]
        # eigenvalues as rayleigh coefficients
        egvalk = φk'*(Hk*φk)
        # compute residual at given kpoint as H(φ)φ - λφ
        res[ik] = Hk*φk - φk*egvalk
    end
    res
end

# To project onto the space tangent to φ, we project δφ onto the orthogonal of φ
function proj_tangent(δφ, φ; tol_test=1e-12)

    Nk1 = size(δφ,1)
    Nk2 = size(φ,1)
    @assert Nk1 == Nk2
    Nk = Nk1

    Πδφ = similar(δφ)

    for ik = 1:Nk
        φk = φ[ik]
        δφk = δφ[ik]
        Πδφk = deepcopy(δφk)

        N1 = size(δφk,2)
        N2 = size(φk,2)
        @assert N1 == N2
        N = N1

        for i = 1:N, j = 1:N
            Πδφk[:,i] -= (φk[:,j]'δφk[:,i]) * φk[:,j]
        end
        Πδφ[ik] = Πδφk
    end

    # test orthogonalisation
    for ik = 1:Nk
        φk = φ[ik]
        δφk = δφ[ik]
        Πδφk = Πδφ[ik]
        N = size(φk,2)
        for i = 1:N, j = 1:N
            @assert abs(Πδφk[:,i]'φk[:,j]) < tol_test [println(abs(Πδφk[:,i]'φk[:,j]))]
        end
    end

    Πδφ
end

# KrylovKit custom orthogonaliser to be used in KrylovKit eigsolve, svdsolve,
# linsolve, ...
# This has to be passed in KrylovKit solvers when studying (Ω+K) as a super
# operator from the tangent space to the tangent space to be sure that
# iterations are constrained to stay on the tangent space.
struct OrthogonalizeAndProject{F, O <: Orthogonalizer, φ} <: Orthogonalizer
    projector::F
    orth::O
    φ::φ
end
OrthogonalizeAndProject(projector, φ) = OrthogonalizeAndProject(projector,
                                                                KrylovDefaults.orth,
                                                                φ)
function KrylovKit.orthogonalize!(v::T, b::OrthonormalBasis{T}, x::AbstractVector,
                                        alg::OrthogonalizeAndProject) where {T}
    v, x = orthogonalize!(v, b, x, alg.orth)
    v = reshape(v, size(alg.φ))
    v = pack(alg.projector(v, alg.φ))::T
    v, x
end
function KrylovKit.orthogonalize!(v::T, x::AbstractVector,
                                        alg::OrthogonalizeAndProject) where {T}
    v, x = orthogonalize!(v, x, alg.orth)
    v = reshape(v, size(alg.φ))
    v = pack(alg.projector(v, alg.φ))::T
    v, x
end
function KrylovKit.gklrecurrence(operator, U::OrthonormalBasis, V::OrthonormalBasis, β,
                                 alg::OrthogonalizeAndProject)
    u = U[end]
    v = operator(u, true)
    v = axpy!(-β, V[end], v)
    # for q in V # not necessary if we definitely reorthogonalize next step and previous step
    #     v, = orthogonalize!(v, q, ModifiedGramSchmidt())
    # end
    α = norm(v)
    rmul!(v, inv(α))

    r = operator(v, false)
    r = axpy!(-α, u, r)
    for q in U
        r, = orthogonalize!(r, q, alg)
    end
    β = norm(r)
    return v, r, α, β
end

############################# OPERATORS ########################################

"""
    apply_Ω(basis::PlaneWaveBasis, δφ, φ, H, λ)

Compute the application of Ω to an element in the space tangent to φ.
It can be done analitically with the formula
      Ωδφi = H*δφi - Σj <δφj|H|δφi> δφj - Σj λji δφj
where λij = <φi|H|φj>;
"""
function apply_Ω(basis::PlaneWaveBasis, δφ, φ, H)
    Nk = length(basis.kpoints)
    Ωδφ = similar(φ)

    for ik = 1:Nk
        φk = φ[ik]
        δφk = δφ[ik]
        Hk = H.blocks[ik]
        λk = φk'*(Hk*φk)
        Ωδφk = similar(δφk)

        N1 = size(δφk,2)
        N2 = size(φk,2)
        @assert N1 == N2
        N = N1

        for i = 1:N
            Hδφki = Hk * δφk[:,i]
            Ωδφk[:,i] = Hδφki
            for j = 1:N
                Ωδφk[:,i] -= (φk[:,j]'Hδφki) * φk[:,j]
                Ωδφk[:,i] -= λk[j,i] * δφk[:,j]
            end
        end
        Ωδφ[ik] = Ωδφk
    end
    # ensure proper projection onto the tangent space
    proj_tangent(Ωδφ, φ)
end

"""
    apply_K(basis::PlaneWaveBasis, δφ, φ, ρ, occ)

Compute the application of K to an element in the space tangent to φ
with the same notations as for the computation of Ω, we have
      Kδφi = Π(dV * φi),
Π being the projection on the space tangent to the φ and
      dV = kernel(δρ)
where δρ = Σi φi*conj(δφi) + hc.
"""
@views function apply_K(basis::PlaneWaveBasis, δφ, φ, ρ, occ)
    Nk = length(basis.kpoints)
    n_spin = basis.model.n_spin_components

    ## compute δρ = Σi φi*conj(δφi)
    δρ_fourier = zeros(complex(eltype(ρ)), size(ρ)...)
    for (ik, kpt) in enumerate(basis.kpoints)
        δρk = zeros(eltype(ρ), basis.fft_size)
        for i = 1:size(φ[ik], 2)
            φki_real = G_to_r(basis, kpt, φ[ik][:,i])
            δφki_real = G_to_r(basis, kpt, δφ[ik][:,i])
            δρk .+= occ[ik][i] .* (conj.(φki_real) .* δφki_real +
                                   conj.(δφki_real) .* φki_real)
        end
        # Check sanity in the density, we should have ∫δρ = 0
        T = real(eltype(δρk))
        sum_δρ = sum(δρk)
        if abs(sum_δρ) > sqrt(eps(T))
            @warn("Mismatch in δρ", sum_δρ=sum_δρ)
        end
        δρk_fourier = r_to_G(basis, complex(δρk))
        lowpass_for_symmetry!(δρk_fourier, basis)
        accumulate_over_symmetries!(δρ_fourier[:, :, :, kpt.spin], δρk_fourier,
                                    basis, basis.ksymops[ik])
    end
    mpi_sum!(δρ_fourier, basis.comm_kpts)
    count = sum(length(basis.ksymops[ik]) for ik in 1:length(basis.kpoints)) ÷ n_spin
    count = mpi_sum(count, basis.comm_kpts)
    δρ = G_to_r(basis, δρ_fourier) ./ count

    ## dV = kernel(δρ)
    dV = apply_kernel(basis, δρ; ρ=ρ)

    ## Kδφi = Π(dV * φi)
    Kδφ = similar(φ)
    for ik = 1:Nk
        kpt = basis.kpoints[ik]
        φk = φ[ik]
        dVφk = similar(φk)

        for i = 1:size(φk,2)
            φk_r = G_to_r(basis, kpt, φk[:,i])
            dVφk_r = dV[:, :, :, kpt.spin] .* φk_r
            dVφk[:,i] = r_to_G(basis, kpt, dVφk_r)
        end
        Kδφ[ik] = dVφk
    end
    # ensure proper projection onto the tangent space
    proj_tangent(Kδφ, φ)
end

############################# NEWTON ALGORITHM #################################

"""
    newton_step(basis::PlaneWaveBasis, φ, φproj, res, occ;
                tol_krylov=1e-12, krylov_verbosity=1)

Perform a newton step : we take as given a planewave set φ and we return the
newton step δφ and the updated state φ - δφ
(after proper orthonormalization) where δφ solves Jac * δφ = res
and Jac is the Jacobian of the projected gradient descent : Jac = Ω+K.
δφ is an element of the tangent space at φproj (set to φ if not specified in newton function).
"""
function newton_step(basis::PlaneWaveBasis, φ, φproj, res, occ;
                     tol_krylov=1e-12, krylov_verbosity=1)

    # necessary quantities
    N = size(φ[1],2)
    Nk = length(basis.kpoints)

    # compute quantites at the point which define the tangent space
    ρproj = compute_density(basis, φproj, occ)
    energies, Hproj = energy_hamiltonian(basis, φproj, occ; ρ=ρproj)

    # packing routines
    unpack = unpacking(φ)
    packed_proj(δφ,φ) = proj_tangent(unpack(δφ), unpack(φ))

    # mapping of the linear system on the tangent space
    function f(x)
        δφ = unpack(x)
        δφ = proj_tangent(δφ, φproj)
        Kδφ = apply_K(basis, δφ, φproj, ρproj, occ)
        Ωδφ = apply_Ω(basis, δφ, φproj, Hproj)
        pack(Ωδφ .+ Kδφ)
    end

    # project res on the good tangent space before starting
    res = proj_tangent(res, φproj)

    # solve (Ω+K) δφ = res on the tangent space with KrylovKit
    δφ, info = linsolve(f, pack(res);
                        tol=tol_krylov, verbosity=krylov_verbosity,
                        # important to specify custom orthogonaliser to keep
                        # iterations on the tangent space
                        orth=OrthogonalizeAndProject(packed_proj, pack(φproj)))
    δφ = unpack(δφ)
    # ensure proper projection onto the tangent space
    δφ = proj_tangent(δφ, φ)
    φ_newton = similar(φ)

    # perform newton_step
    for ik = 1:Nk
        φ_newton[ik] = ortho_qr(φ[ik] - δφ[ik])
    end
    (φ_newton, δφ)
end

"""
    newton(basis::PlaneWaveBasis; ψ0=nothing,
           tol=1e-6, max_iter=100, φproj=nothing, verbosity=0)

Newton algorithm. Be careful that the starting needs to be not too far from the solution.
Still contains debugging features.
If φproj is nothing, we use the successive φ's to define the tangent spaces on which we solve
Newton's equation. φproj can be useful when we want to fix the tangent space for testing.
"""
function newton(basis::PlaneWaveBasis{T}; ψ0=nothing,
                tol=1e-6, max_iter=20, φproj=nothing, verbosity=0) where T

    ## setting parameters
    model = basis.model
    @assert model.spin_polarization in (:none, :spinless)
    @assert model.temperature == 0 # temperature is not yet supported
    filled_occ = filled_occupation(model)
    N = div(model.n_electrons, filled_occ)

    ## number of kpoints
    Nk = length(basis.kpoints)
    occupation = [filled_occ * ones(T, N) for ik = 1:Nk]

    ## starting point and keep only occupied orbitals
    if ψ0 === nothing
        ψ0 = [ortho_qr(randn(Complex{T}, length(G_vectors(kpt)), N))
              for kpt in basis.kpoints]
    else
        for ik in 1:Nk
            ψ0[ik] = ψ0[ik][:,1:N]
        end
    end
    if φproj != nothing
        for ik in 1:Nk
            φproj[ik] = φproj[ik][:,1:N]
        end
    end

    ## iterators
    err = 1
    k = 0

    ## orbitals, densities and energies to be updated along the iterations
    φ = deepcopy(ψ0)
    ρ = compute_density(basis, φ, occupation)
    prev_energies = nothing

    @printf "\nn     %-12s      Eₙ-Eₙ₋₁     ρ_next-ρ\n" "Energy"
    @printf "---   ---------------   ---------   --------\n"

    ## perform iterations
    while err > tol && k < max_iter
        k += 1

        # set ϕ which defines the tangent space
        if φproj === nothing
            ϕ = φ
        else
            ϕ = φproj
        end

        # compute next step
        res = compute_residual(basis, φ, occupation)
        φ, δφ = newton_step(basis, φ, ϕ, res, occupation;
                            krylov_verbosity=verbosity)

        # compute error on the densities and the energies
        ρ_next = compute_density(basis, φ, occupation)
        err = norm(ρ_next - ρ)
        energies, H = energy_hamiltonian(basis, φ, occupation; ρ=ρ)
        E = energies.total
        Estr = (@sprintf "%+15.12f" round(E, sigdigits=13))[1:15]
        prev_E = prev_energies === nothing ? Inf : prev_energies.total
        ΔE     = prev_E == Inf ? "      NaN" : @sprintf "% 3.2e" E - prev_E
        @printf "% 3d   %s   %s   %2.2e \n" k Estr ΔE err

        # update
        ρ = ρ_next
        prev_energies = energies
    end

    # Rayleigh-Ritz
    energies, H = energy_hamiltonian(basis, φ, occupation; ρ=ρ)
    eigenvalues = []
    for ik = 1:Nk
        Hφk = H.blocks[ik] * φ[ik]
        F = eigen(Hermitian(φ[ik]'Hφk))
        push!(eigenvalues, F.values)
        φ[ik] .= φ[ik] * F.vectors
    end

    εF = nothing  # does not necessarily make sense here, as the
                  # Aufbau property might not even be true

    # return results
    cvg = k == max_iter ? false : true
    (ham=H, basis=basis, energies=energies, converged=cvg,
     ρ=ρ, ψ=φ, eigenvalues=eigenvalues, occupation=occupation, εF=εF)
end
