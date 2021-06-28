# Newton's algorithm to solve SCF equations
#
# Newton algorithm consist of iterating over density matrices like
#       P <- P - δP, with additional renormalization
# where δP solves ([1])
#       (Ω+K)δP = [P, [P, H(P)]]
# where (Ω+K) if the constrained Hessian of the energy. It is
# defined as a super operator on the space tangent to the constraints
# manifold at point P_∞, the solution of our problem.
#   - K is the unconstrained Hessian of the energy on the manifold;
#   - Ω represents the influence of the curvature of the manifold :
#         ΩδP = -[P_∞, [H(P_∞), δP]].
#     In practice, we dont have access to P_∞ so we just use the current P.
#     Another way to see Ω is as the four-point independent-particle
#     susceptibility. Indeed, if we define the following extension and
#     retraction operators
#       - E : builds a density matrix Eρ given a density ρ via Eρ(r,r') = δ(r,r')ρ(r),
#       - R : builds a density ρ given a density matrix O via ρ(r) = O(r,r),
#     then we have the relation χ0_2P = -R(Ω_4P^-1)E, where χ0_2P is the two_point
#     independent-particle susceptibility (it returns δρ from a given δV) and
#     Ω_4P is the four-point independent-particle susceptibility.
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
#         Ωδψi = H*δψi - Σj (<ψj|H|δψi> ψj + <ψj|δψi> Hψj)/2 - Σj λji δψj
#     where λij = <ψi|H|ψj>. This is one way among others to extend the
#     definition of ΩδP = -[P_∞, [H(P_∞), δP]] to any point P on the manifold --
#     we chose this one because it makes Ω self-adjoint for the Frobenius scalar
#     product;
#   - computing Kδψ can be done by constructing the δρ associated with δψ, then
#     using the exchange-correlation kernel to compute the associated δV, then
#     acting with δV on the ψ and projecting on the tangent space.

using LinearMaps
using IterativeSolvers

#  Compute the gradient of the energy, projected on the space tangent to ψ, that
#  is to say H(ψ)*ψ - ψ*λ where λ is the set of Rayleigh coefficients associated
#  to the ψ.
function compute_projected_gradient(basis::PlaneWaveBasis, ψ, occupation)
    ρ = compute_density(basis, ψ, occupation)
    _, H = energy_hamiltonian(basis, ψ, occupation; ρ=ρ)

    [proj_tangent_kpt(H.blocks[ik] * ψ[ik], ψ[ik])
     for ik = 1:length(basis.kpoints)]
end

# Projections on the space tangent to ψ
function proj_tangent_kpt!(δψk, ψk)
    # δψk <- δψk - ψk * (ψk'δψk)
    mul!(δψk, ψk, ψk'δψk, -1, 1)
end
proj_tangent_kpt(δψk, ψk) = proj_tangent_kpt!(copy(δψk), ψk)
function proj_tangent(δψ, ψ)
    [proj_tangent_kpt(δψ[ik], ψ[ik]) for ik = 1:size(ψ,1)]
end
function proj_tangent!(δψ, ψ)
    [proj_tangent_kpt!(δψ[ik], ψ[ik]) for ik = 1:size(ψ,1)]
end

function apply_Ω(basis::PlaneWaveBasis, δψ, ψ, H::Hamiltonian)

    δψ = proj_tangent(δψ, ψ)
    Ωδψ = map(1:length(basis.kpoints)) do ik
        ψk = ψ[ik]
        Hk = H.blocks[ik]
        δψk = δψ[ik]

        Hψk = Hk * ψk
        ψHψk = Hermitian(ψk * Hψk')
        λk = ψk'Hψk

        Ωδψk = Hk * δψk - ψHψk * δψk
        Ωδψk .-= δψk * λk
    end

    proj_tangent!(Ωδψ, ψ)
end

# The variation in density corresponds to a variation in the orbitals.
@views function compute_δρ(basis::PlaneWaveBasis{T}, ψ, δψ,
                           occupation, δoccupation=0 .* occupation) where T
    n_spin = basis.model.n_spin_components

    δρ_fourier = zeros(complex(T), basis.fft_size..., n_spin)
    for (ik, kpt) in enumerate(basis.kpoints)
        δρk = zeros(T, basis.fft_size)
        for n = 1:size(ψ[ik], 2)
            ψnk_real = G_to_r(basis, kpt, ψ[ik][:, n])
            δψnk_real = G_to_r(basis, kpt, δψ[ik][:, n])
            δρk .+= occupation[ik][n] .* (conj.(ψnk_real) .* δψnk_real +
                                          conj.(δψnk_real) .* ψnk_real)
            δρk .+= δoccupation[ik][n] .* abs2.(ψnk_real)
        end
        # Check sanity in the density, we should have ∫δρ = 0
        if abs(sum(δρk)) > sqrt(eps(T))
            @warn("Mismatch in δρ", sum_δρ=sum(δρk))
        end
        δρk_fourier = r_to_G(basis, complex(δρk))
        lowpass_for_symmetry!(δρk_fourier, basis)
        accumulate_over_symmetries!(δρ_fourier[:, :, :, kpt.spin], δρk_fourier,
                                    basis, basis.ksymops[ik])
    end
    mpi_sum!(δρ_fourier, basis.comm_kpts)
    count = sum(length(basis.ksymops[ik]) for ik in 1:length(basis.kpoints)) ÷ n_spin
    count = mpi_sum(count, basis.comm_kpts)
    G_to_r(basis, δρ_fourier) ./ count
end

@views function apply_K(basis::PlaneWaveBasis, δψ, ψ, ρ, occupation)

    δψ = proj_tangent(δψ, ψ)
    δρ = compute_δρ(basis, ψ, δψ, occupation)
    δV = apply_kernel(basis, δρ; ρ=ρ)

    Kδψ = map(1:length(basis.kpoints)) do ik
        kpt = basis.kpoints[ik]
        ψk = ψ[ik]
        δVψk = similar(ψk)

        for n = 1:size(ψk, 2)
            ψnk_real = G_to_r(basis, kpt, ψk[:,n])
            δVψnk_real = δV[:, :, :, kpt.spin] .* ψnk_real
            δVψk[:, n] = r_to_G(basis, kpt, δVψnk_real)
        end
        δVψk
    end
    # ensure projection onto the tangent space
    proj_tangent!(Kδψ, ψ)
end

function NewtonConvergenceDensity(tolerance)
    info -> (norm(info.ρ_next - info.ρ) * sqrt(info.basis.dvol) < tolerance)
end

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

"""
    solve_ΩplusK(basis::PlaneWaveBasis{T}, ψ, res, occupation;
                 tol_cg=1e-10, verbose=false) where T

Return δψ where (Ω+K) δψ = rhs
"""
function solve_ΩplusK(basis::PlaneWaveBasis{T}, ψ, rhs, occupation;
                     tol_cg=1e-10, verbose=false) where T

    Nk = length(basis.kpoints)

    # compute quantites at the point which define the tangent space
    ρ = compute_density(basis, ψ, occupation)
    _, H = energy_hamiltonian(basis, ψ, occupation; ρ=ρ)

    # pack and unpack
    pack(ψ) = pack_ψ(basis, ψ)
    unpack(x) = unpack_ψ(basis, x)

    # project rhs on the tangent space before starting
    proj_tangent!(rhs, ψ)
    rhs_pack = pack(rhs)

    # preconditioner
    Pks = [PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
    for ik = 1:length(Pks)
        precondprep!(Pks[ik], ψ[ik])
    end
    function f_ldiv!(x, y)
        δψ = unpack(y)
        proj_tangent!(δψ, ψ)
        Pδψ = [ Pks[ik] \ δψk for (ik, δψk) in enumerate(δψ)]
        proj_tangent!(Pδψ, ψ)
        x .= pack(Pδψ)
    end

    # mapping of the linear system on the tangent space
    function ΩpK(x)
        δψ = unpack(x)
        Kδψ = apply_K(basis, δψ, ψ, ρ, occupation)
        Ωδψ = apply_Ω(basis, δψ, ψ, H)
        pack(Ωδψ + Kδψ)
    end
    J = LinearMap{T}(ΩpK, size(rhs_pack, 1))

    # solve (Ω+K) δψ = rhs on the tangent space with CG
    δψ = cg(J, rhs_pack, Pl=FunctionPreconditioner(f_ldiv!),
            reltol=tol_cg/norm(rhs), verbose=verbose)

    unpack(δψ)
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
                callback=NewtonDefaultCallback(),
                is_converged=NewtonConvergenceDensity(tol)) where T

    # setting parameters
    model = basis.model
    @assert model.spin_polarization in (:none, :spinless, :collinear)
    @assert model.temperature == 0 # temperature is not yet supported
    filled_occ = filled_occupation(model)
    n_spin = basis.model.n_spin_components
    n_bands = div(div(model.n_electrons, filled_occ), n_spin)

    # number of kpoints and occupation
    Nk = length(basis.kpoints)
    occupation = [filled_occ * ones(T, n_bands) for ik = 1:Nk]

    # check that there are no virtual orbitals
    @assert n_bands == size(ψ0[1],2)

    # iterators
    err = Inf
    n_iter = 0

    # orbitals, densities and energies to be updated along the iterations
    ψ = deepcopy(ψ0)
    ρ = compute_density(basis, ψ, occupation)
    energies, H = energy_hamiltonian(basis, ψ, occupation; ρ=ρ)
    converged = false

    # perform iterations
    while !converged && n_iter < maxiter
        n_iter += 1

        # compute next step
        res = compute_projected_gradient(basis, ψ, occupation)
        # solve (Ω+K) δψ = -res so that the Newton step is ψ <- ψ+δψ
        δψ = solve_ΩplusK(basis, ψ, -res, occupation; tol_cg=tol_cg,
                          verbose=verbose)
        ψ = map(1:Nk) do ik ortho_qr(ψ[ik] + δψ[ik]) end

        # callback
        ρ_next = compute_density(basis, ψ, occupation)
        energies, H = energy_hamiltonian(basis, ψ, occupation; ρ=ρ)
        info = (ham=H, basis=basis, converged=converged, stage=:iterate,
                ρ=ρ, ρ_next=ρ_next, n_iter=n_iter, energies=energies)
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
            n_iter=n_iter, ψ=ψ, stage=:finalize)
    callback(info)
    info
end
