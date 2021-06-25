# Newton's algorithm to solve SCF equations
#
# Newton algorithm consist of iterating over density matrices like
#       P = P - δP, with proper renormalization
# where δP solves
#       (Ω+K)δP = [P, [P, H(P)]] (residual)
# where (Ω+K) if the constrained Hessian of the energy. It is
# defined as a super operator on the space tangent to the constraints
# manifold at point P_∞, the solution of our problem.
#   - K is the unconstrained Hessian of the energy on the manifold;
#   - Ω represents the influence of the curvature of the manifold :
#         ΩδP = -[P_∞, [H(P_∞), δP]].
#     In practice, we dont have access to P_∞ so we just use the current P.
#     Another way to see Ω is to define the following extension and retraction operators
#       - E : builds a density matrix Eρ given a density ρ via Eρ(r,r') = δ(r,r')ρ(r),
#       - R : builds a density ρ given a density matrix O via ρ(r) = O(r,r),
#     then we have the relation χ0 = -R(Ω^-1)E, where χ0 is the independent-particle
#     susceptibility and returns δρ from a given δV.
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
#     where λij = <ψi|H|ψj>. This is one way among other to extend the
#     definition of ΩδP = -[P_∞, [H(P_∞), δP]] to any point P on the manifold --
#     we chose this one because it is self-adjoint;
#   - computing Kδψ can be done by constructing the dρ associated with dψ, then
#     using the exchange-correlation kernel to compute the associated dV, then
#     acting with dV on the ψ and projecting on the tangent space;
#   - to perform Newton iterations, we solve (Ω+K)δψ = Hψ-λψ and then perform
#     ψ = ψ - δψ, with proper orthonormalization.
#
# For further details :
# Eric Cancès, Gaspard Kemlin, Antoine Levitt. Convergence analysis of
# direct minimization and self-consistent iterations. SIAM Journal of Matrix
# Analysis and Applications, 42(1):243–274 (2021)

using LinearMaps
using IterativeSolvers

# Tools

"""
    compute_projected_gradient(basis::PlaneWaveBasis, ψ, occupation)

Compute the residual associated to a set of planewave ψ, that is to say
H(ψ)*ψ - ψ*λ where λ is the set of Rayleigh coefficients associated to the ψ.
"""
function compute_projected_gradient(basis::PlaneWaveBasis, ψ, occupation)
    Nk = length(basis.kpoints)
    ρ = compute_density(basis, ψ, occupation)
    _, H = energy_hamiltonian(basis, ψ, occupation; ρ=ρ)

    [proj_tangent_kpt(H.blocks[ik]*ψ[ik], ψ[ik]) for ik = 1:Nk]
end

"""
    proj_tangent_kpt(δψk, ψk)

Computation the projection of δψk onto the tangent space defined at ψk, where
δψk and ψk are defined at the same kpoint.
"""
proj_tangent_kpt(δψk, ψk) = δψk - ψk * (ψk'δψk)
function proj_tangent_kpt!(δψk, ψk)
    mul!(δψk, ψk, ψk'δψk, -1, 1)
end
"""
    proj_tangent(δψ, ψ)

Computation the projection of δψ onto the tangent space defined at ψ.
"""
function proj_tangent(δψ, ψ)
    [proj_tangent_kpt(δψ[ik], ψ[ik]) for ik = 1:size(ψ,1)]
end
function proj_tangent!(δψ, ψ)
    [proj_tangent_kpt!(δψ[ik], ψ[ik]) for ik = 1:size(ψ,1)]
end

# Operators

"""
    apply_Ω(basis::PlaneWaveBasis, δψ, ψ, H)

Apply Ω to an element in the tangent space to ψ.
"""
function apply_Ω(basis::PlaneWaveBasis, δψ, ψ, H)
    Nk = length(basis.kpoints)

    Hψ = [H.blocks[ik] * ψ[ik] for ik = 1:Nk]
    ψHψ = [(ψ[ik]*Hψ[ik]' + Hψ[ik]*ψ[ik]') / 2 for ik = 1:Nk]
    λ = [ψ[ik]'Hψ[ik] for ik = 1:Nk]

    δψ = proj_tangent(δψ, ψ)
    Ωδψ = [H.blocks[ik] * δψ[ik] - ψHψ[ik] * δψ[ik] for ik = 1:Nk]
    Ωδψ = Ωδψ - [δψ[ik]*λ[ik] for ik = 1:Nk]
    proj_tangent!(Ωδψ, ψ)
end

"""
    compute_dρ(ψ, δψ)

Compute δρ = Σi ψi*conj(δψi) + conj(ψi)*δψi
"""
@views function compute_dρ(basis::PlaneWaveBasis, ψ, δψ, ρ, occupation)
                           n_spin = basis.model.n_spin_components

    δρ_fourier = zeros(complex(eltype(ρ)), size(ρ)...)
    for (ik, kpt) in enumerate(basis.kpoints)
        δρk = zeros(eltype(ρ), basis.fft_size)
        for i = 1:size(ψ[ik], 2)
            ψki_real = G_to_r(basis, kpt, ψ[ik][:, i])
            δψki_real = G_to_r(basis, kpt, δψ[ik][:, i])
            δρk .+= occupation[ik][i] .* (conj.(ψki_real) .* δψki_real +
                                          conj.(δψki_real) .* ψki_real)
        end
        # Check sanity in the density, we should have ∫δρ = 0
        T = real(eltype(δρk))
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

"""
    apply_K(basis::PlaneWaveBasis, δψ, ψ, ρ, occupation)

Compute the application of K to an element in the space tangent to ψ.
"""
function apply_K(basis::PlaneWaveBasis, δψ, ψ, ρ, occupation)
    Nk = length(basis.kpoints)

    δψ = proj_tangent(δψ, ψ)
    dρ = compute_dρ(basis, ψ, δψ, ρ, occupation)
    dV = apply_kernel(basis, dρ; ρ=ρ)

    Kδψ = []
    for ik = 1:Nk
        kpt = basis.kpoints[ik]
        ψk = ψ[ik]
        dVψk = similar(ψk)

        for i = 1:size(ψk, 2)
            ψk_r = G_to_r(basis, kpt, ψk[:,i])
            dVψk_r = dV[:, :, :, kpt.spin] .* ψk_r
            dVψk[:, i] = r_to_G(basis, kpt, dVψk_r)
        end
        push!(Kδψ, dVψk)
    end
    # ensure proper projection onto the tangent space
    proj_tangent!(Kδψ, ψ)
end

# Callback and convergence

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

# Newton algorithm

"""
    newton_step(basis::PlaneWaveBasis, ψ, res, occupation;
                tol_cg=1e-10, verbose=false)

Perform a Newton step : we take as given a planewave set ψ and we return the
Newton step δψ and the updated state ψ - δψ
(after proper orthonormalization) where δψ solves Jac * δψ = res
and Jac is the Jacobian of the projected gradient descent : Jac = Ω+K.
δψ is an element of the tangent space at ψ (set to ψ if not specified in newton function).
"""
function newton_step(basis::PlaneWaveBasis, ψ, res, occupation;
                     tol_cg=1e-10, verbose=false)

    N = size(ψ[1], 2)
    Nk = length(basis.kpoints)

    # compute quantites at the point which define the tangent space
    ρ = compute_density(basis, ψ, occupation)
    energies, H = energy_hamiltonian(basis, ψ, occupation; ρ=ρ)

    # packing routines
    # some care is needed here : K is real-linear but not complex-linear because
    # of the computation of δρ from δψ. To overcome this difficulty, instead of
    # seeing the jacobian as an operator from C^Nb to C^Nb, we see it as an
    # operator from R^2Nb to R^2Nb. In practice, this is done with the
    # reinterpret function from julia. Thus, we are sure that the map we define
    # below apply_jacobian is self-adjoint.
    T = eltype(basis)
    pack(ψ) = Array(reinterpret(T, pack_ψ(basis, ψ)))
    unpack(x) = unpack_ψ(basis, reinterpret(Complex{T}, x))

    # project res on the good tangent space before starting
    proj_tangent!(res, ψ)
    rhs = pack(res)

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
    function apply_jacobian(x)
        δψ = unpack(x)
        Kδψ = apply_K(basis, δψ, ψ, ρ, occupation)
        Ωδψ = apply_Ω(basis, δψ, ψ, H)
        pack(Ωδψ + Kδψ)
    end
    J = LinearMap{T}(apply_jacobian, size(rhs, 1))

    # solve (Ω+K) δψ = res on the tangent space with CG
    δψ = cg(J, rhs, Pl=FunctionPreconditioner(f_ldiv!),
            reltol=tol_cg/norm(rhs), verbose=verbose)

    unpack(δψ)
end

"""
    newton(basis::PlaneWaveBasis{T}; ψ0=nothing,
           tol=1e-6, tol_cg=1e-10, maxiter=20, verbose=false,
           callback=NewtonDefaultCallback(),
           is_converged=NewtonConvergenceDensity(tol))

Newton algorithm. Be careful that the starting needs to be not too far from the solution.
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
    N = div(div(model.n_electrons, filled_occ), n_spin)

    # number of kpoints and occupation
    Nk = length(basis.kpoints)
    occupation = [filled_occ * ones(T, N) for ik = 1:Nk]

    # check that there are no virtual orbitals
    @assert N == size(ψ0[1],2)

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
        δψ = newton_step(basis, ψ, res, occupation; tol_cg=tol_cg,
                         verbose=verbose)
        ψ = [ortho_qr(ψ[ik] - δψ[ik]) for ik = 1:Nk]

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
