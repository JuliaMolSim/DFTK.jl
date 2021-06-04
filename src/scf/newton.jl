# Newton's algorithm for the direct minimization of the energy
# ------------------------------------------------------------
#
# Newton algorithm consist of iterating over density matrices like
#       P = P - δP
# where δP solves
#       (Ω+K)δP = [P, [P, H(P)]] (residual)
# where (Ω+K) if the Jacobian of the direct minimization iterations. It is
# defined as a super operator from the space tangent to the constraints
# manifold at point P_∞, the solution of our problem.
#   - K is the Hessian of the energy on the manifold;
#   - Ω represents the influence of the curvature of the manifold :
#         ΩδP = -[P_∞, [H(P_∞), δP]].
#     In practice, we dont have access to P_∞ so we just use the current P.
#     Another way to see Ω is to define the following operators
#       - E : builds a density matrix Eρ given a density ρ via Eρ(r,r') = δ(r,r')ρ(r),
#       - R : builds a density ρ given a density matrix O via ρ(r) = O(r,r),
#     then you have the relation χ0 = -R(Ω^-1)E, where χ0 is the susceptibility
#     and returns δρ from a given δV.
#
# Here, in the orbital framework, an element on the tangent space at φ can be written as
#       δP = Σ |φi><δφi| + hc
# where the δφi are of size Nb and are all orthogonal to the φj, 1 <= j <= N.
# Therefore we store them in the same kind of array than φ, with
# δφ[ik][:, i] = δφi for each k-point.
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

import KrylovKit: Orthogonalizer, OrthonormalBasis, KrylovDefaults, orthogonalize!
using KrylovKit

#
# Tools
#

"""
    compute_scf_residual(basis::PlaneWaveBasis, φ, occ)

Compute the residual associated to a set of planewave φ, that is to say
H(φ)*φ - λ*φ where λ is the set of rayleigh coefficients associated to the φ.
"""
function compute_scf_residual(basis::PlaneWaveBasis, φ, occ)
    Nk = length(basis.kpoints)
    ρ = compute_density(basis, φ, occ)
    _, H = energy_hamiltonian(basis, φ, occ; ρ=ρ)

    # compute residual
    res = similar(φ)
    for (ik, φk) in enumerate(φ)
        Hk = H.blocks[ik]
        # eigenvalues as rayleigh coefficients
        egvalk = φk'*(Hk*φk)
        # compute residual at given kpoint as H(φ)φ - λφ
        res[ik] = Hk*φk - φk*egvalk
    end
    res
end

"""
    proj_tangent(δφ, φ; tol_test=1e-12)

Computation the projection of δφ onto the tangent space defined at φ.
"""
function proj_tangent(δφ, φ)
    @assert size(δφ, 1) == size(φ, 1)
    Nk = size(φ, 1)

    # compute projection
    Πδφ = similar(δφ)
    for (ik, φk) in enumerate(φ)
        δφk = δφ[ik]
        Πδφk = deepcopy(δφk)

        @assert size(δφk, 2) == size(φk, 2)
        N = size(φk, 2)

        for i = 1:N, j = 1:N
            Πδφk[:, i] -= (φk[:, j]'δφk[:, i]) * φk[:, j]
        end
        Πδφ[ik] = Πδφk
    end

    # test orthogonalisation
    tol_test = eps(real(eltype(φ[1])))^(2/3)
    for (ik, φk) in enumerate(φ)
        δφk = δφ[ik]
        Πδφk = Πδφ[ik]
        N = size(φk, 2)
        for i = 1:N, j = 1:N
            if abs(Πδφk[:, i]'φk[:, j]) > tol_test
                @warn "Projection failed : <Πδφki,φkj> = $(abs(Πδφk[:, i]'φk[:, j]))"
            end
        end
    end

    Πδφ
end

# KrylovKit custom orthogonaliser to be used in KrylovKit eigsolve, svdsolve,
# linsolve, ...
# This has to be passed in KrylovKit solvers when studying (Ω+K) as a super
# operator from the tangent space to the tangent space to be sure that
# iterations are constrained to stay on the tangent space.
# /!\ Be careful that here, φproj needs to be packed into on big array
struct OrthogonalizeAndProject{F, O <: Orthogonalizer, φproj} <: Orthogonalizer
    projector::F
    orth::O
    φ::φproj
end
OrthogonalizeAndProject(projector, φ) = OrthogonalizeAndProject(projector,
                                                                KrylovDefaults.orth,
                                                                φ)
function KrylovKit.orthogonalize!(v::T, b::OrthonormalBasis{T}, x::AbstractVector,
                                        alg::OrthogonalizeAndProject) where {T}
    v, x = orthogonalize!(v, b, x, alg.orth)
    v = reshape(v, size(alg.φ))
    v = alg.projector(v, alg.φ)::T
    v, x
end
function KrylovKit.orthogonalize!(v::T, x::AbstractVector,
                                        alg::OrthogonalizeAndProject) where {T}
    v, x = orthogonalize!(v, x, alg.orth)
    v = reshape(v, size(alg.φ))
    v = alg.projector(v, alg.φ)::T
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

#
# Operators
#

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
        λk = φk' * (Hk*φk)
        Ωδφk = similar(δφk)

        N1 = size(δφk, 2)
        N2 = size(φk, 2)
        @assert N1 == N2
        N = N1

        for i = 1:N
            Hδφki = Hk * δφk[:, i]
            Ωδφk[:, i] = Hδφki
            for j = 1:N
                Ωδφk[:, i] -= (φk[:, j]'Hδφki) * φk[:, j]
                Ωδφk[:, i] -= λk[j, i] * δφk[:, j]
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
            φki_real = G_to_r(basis, kpt, φ[ik][:, i])
            δφki_real = G_to_r(basis, kpt, δφ[ik][:, i])
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

        for i = 1:size(φk, 2)
            φk_r = G_to_r(basis, kpt, φk[:,i])
            dVφk_r = dV[:, :, :, kpt.spin] .* φk_r
            dVφk[:, i] = r_to_G(basis, kpt, dVφk_r)
        end
        Kδφ[ik] = dVφk
    end
    # ensure proper projection onto the tangent space
    proj_tangent(Kδφ, φ)
end

#
# Callback and convergence
#

"""
Flag convergence as soon as total energy change drops below tolerance
"""
function NewtonConvergenceEnergy(tolerance)
    energy_total = NaN

    function is_converged(info)
        info.energies === nothing && return false # first iteration

        etot_old = energy_total
        energy_total = info.energies.total
        abs(energy_total - etot_old) < tolerance
    end
    return is_converged
end
:w

"""
Flag convergence as soon as density change drops below tolerance
"""
function NewtonConvergenceDensity(tolerance)
    info -> (norm(info.ρ_next - info.ρ) * sqrt(info.basis.dvol) < tolerance)
end

"""
Default callback function for `newton`, which prints a convergence table
"""
function NewtonDefaultCallback()
    prev_energies = nothing
    function callback(info)
        !mpi_master() && return info  # Printing only on master
        if info.stage == :finalize
            info.converged || @warn "Newton not converged."
            return info
        end
        collinear = info.basis.model.spin_polarization == :collinear

        if info.n_iter == 1
            E_label = haskey(info.energies, "Entropy") ? "Free energy" : "Energy"
            magn    = collinear ? ("   Magnet", "   ------") : ("", "")
            @printf "n     %-12s      Eₙ-Eₙ₋₁     ρ_next-ρ%s\n" E_label magn[1]
            @printf "---   ---------------   ---------   --------%s\n" magn[2]
        end
        E    = isnothing(info.energies) ? Inf : info.energies.total
        Δρ   = norm(info.ρ_next - info.ρ) * sqrt(info.basis.dvol)
        if size(info.ρ_next, 4) == 1
            magn = NaN
        else
            magn = sum(spin_density(info.ρ_next)) * info.basis.dvol
        end

        Estr   = (@sprintf "%+15.12f" round(E, sigdigits=13))[1:15]
        prev_E = prev_energies === nothing ? Inf : prev_energies.total
        ΔE     = prev_E == Inf ? "      NaN" : @sprintf "% 3.2e" E - prev_E
        Mstr = collinear ? "   $((@sprintf "%6.3f" round(magn, sigdigits=4))[1:6])" : ""
        @printf "% 3d   %s   %s   %2.2e%s\n" info.n_iter Estr ΔE Δρ Mstr
        prev_energies = info.energies

        flush(stdout)
        info
    end
end

#
# Newton algorithm
#

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
    N = size(φ[1], 2)
    Nk = length(basis.kpoints)

    # compute quantites at the point which define the tangent space
    ρproj = compute_density(basis, φproj, occ)
    energies, Hproj = energy_hamiltonian(basis, φproj, occ; ρ=ρproj)

    # packing routines
    pack(φ) = pack_arrays(basis, φ)
    unpack(x) = unpack_arrays(basis, x)
    packed_proj(δx, x) = pack(proj_tangent(unpack(δx), unpack(x)))

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
    newton(basis::PlaneWaveBasis{T}; ψ0=nothing,
                    tol=1e-6, maxiter=20, φproj=nothing, verbosity=0,
                    callback=NewtonDefaultCallback(),
                    is_converged=NewtonConvergenceDensity(tol))

Newton algorithm. Be careful that the starting needs to be not too far from the solution.
Still contains debugging features.
If φproj is nothing, we use the successive φ's to define the tangent spaces on which we solve
Newton's equation. φproj can be useful when we want to fix the tangent space for testing.
"""
function newton(basis::PlaneWaveBasis{T}; ψ0=nothing,
                tol=1e-6, maxiter=20, φproj=nothing, verbosity=0,
                callback=NewtonDefaultCallback(),
                is_converged=NewtonConvergenceDensity(tol)) where T

    ## setting parameters
    model = basis.model
    @assert model.spin_polarization in (:none, :spinless, :collinear)
    @assert model.temperature == 0 # temperature is not yet supported
    filled_occ = filled_occupation(model)
    n_spin = basis.model.n_spin_components
    N = div(div(model.n_electrons, filled_occ), n_spin)

    ## number of kpoints and occupation
    Nk = length(basis.kpoints)
    occupation = [filled_occ * ones(T, N) for ik = 1:Nk]

    ## starting point and keep only occupied orbitals
    if ψ0 === nothing
        ψ0 = [ortho_qr(randn(Complex{T}, length(G_vectors(kpt)), N))
              for kpt in basis.kpoints]
    else
        for ik in 1:Nk
            ψ0[ik] = ψ0[ik][:, 1:N]
        end
    end
    update_φproj = true
    if φproj != nothing
        update_φproj = false
        for ik in 1:Nk
            φproj[ik] = φproj[ik][:, 1:N]
        end
    end

    ## iterators
    err = 1
    n_iter = 0

    ## orbitals, densities and energies to be updated along the iterations
    φ = deepcopy(ψ0)
    ρ = compute_density(basis, φ, occupation)
    prev_energies = nothing
    converged = false

    ## perform iterations
    while !converged && n_iter < maxiter
        n_iter += 1

        # update φproj which defines the tangent space if necessary
        if update_φproj
            φproj = φ
        end

        # compute next step
        res = compute_scf_residual(basis, φ, occupation)
        φ, δφ = newton_step(basis, φ, φproj, res, occupation;
                            krylov_verbosity=verbosity)

        # callback
        ρ_next = compute_density(basis, φ, occupation)
        energies, H = energy_hamiltonian(basis, φ, occupation; ρ=ρ)
        info = (ham=H, basis=basis, converged=converged, stage=:iterate,
                ρ=ρ, ρ_next=ρ_next, n_iter=n_iter, energies=energies)
        callback(info)

        # update and test convergence
        converged = is_converged(info)
        ρ = ρ_next
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

    # return results and call callback one last time with final state for clean
    # up
    info = (ham=H, basis=basis, energies=energies, converged=converged,
            ρ=ρ, eigenvalues=eigenvalues, occupation=occupation, εF=εF,
            n_iter=n_iter, ψ=φ, stage=:finalize)
    callback(info)
    info
end
