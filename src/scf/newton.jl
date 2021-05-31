# Newton's algorithm for the direct minimization of the energy

"""
    newton_step(basis::PlaneWaveBasis, φ, φproj, res, occ;
                tol_krylov=1e-12, krylov_verb=1)

Perform a newton step : we take as given a planewave set φ and we return the
newton step δφ and the updated state φ - δφ
(after proper orthonormalization) where δφ solves Jac * δφ = res
and Jac is the Jacobian of the projected gradient descent : Jac = Ω+K.
(cf. Eric Cancès, Gaspard Kemlin, Antoine Levitt. Convergence analysis of
direct minimization and self-consistent iterations. SIAM Journal of Matrix
Analysis and Applications, 42(1):243–274 (2021))
"""
function newton_step(basis::PlaneWaveBasis, φ, φproj, res, occ;
                     tol_krylov=1e-12, krylov_verb=1)

    # necessary quantities
    N = size(φ[1],2)
    Nk = length(basis.kpoints)
    ortho(ψk) = Matrix(qr(ψk).Q)

    # compute quantites at the point which define the tangent space
    ρproj = compute_density(basis, φproj, occ)
    energies, Hproj = energy_hamiltonian(basis, φproj, occ; ρ=ρproj)
    λproj = similar(φ)
    for ik = 1:Nk
        φk = φ[ik]
        Hk = Hproj.blocks[ik]
        λproj[ik] = φk'*(Hk*φk)
    end

    # packing routines
    pack, unpack, packed_proj = packing(basis, φproj)

    # solve linear system with KrylovKit
    function f(x)
        δφ = unpack(x)
        ΩpKx = ΩplusK(basis, δφ, φproj, ρproj, Hproj, λproj, occ)
        pack(ΩpKx)
    end

    # project res on the good tangent space before starting
    res = proj(res, φproj)

    # solve (Ω+K) δφ = res
    δφ, info = linsolve(f, pack(res);
                        tol=tol_krylov, verbosity=krylov_verb,
                        orth=OrthogonalizeAndProject(packed_proj, pack(φproj)))
    δφ = unpack(δφ)
    # ensure proper projection onto the tangent space
    δφ = proj(δφ, φ)
    φ_newton = similar(φ)

    # perform newton_step
    for ik = 1:Nk
        φ_newton[ik] = ortho(φ[ik] - δφ[ik])
    end
    (φ_newton, δφ)
end

"""
    newton(basis::PlaneWaveBasis; ψ0=nothing,
           tol=1e-6, max_iter=100, φproj=nothing, verb=1)

Newton algorithm. Be careful that the starting needs to be not too far from the solution.
Still contains debugging features.
"""
function newton(basis::PlaneWaveBasis{T}; ψ0=nothing,
                tol=1e-6, max_iter=100, φproj=nothing, verb=0) where T

    ## setting parameters
    model = basis.model
    @assert model.spin_polarization in (:none, :spinless)
    @assert model.temperature == 0 # temperature is not yet supported
    filled_occ = filled_occupation(model)
    N = div(model.n_electrons, filled_occ)

    ## number of kpoints
    Nk = length(basis.kpoints)
    occupation = [filled_occ * ones(T, N) for ik = 1:Nk]

    ## starting point
    if ψ0 === nothing
        ortho(ψk) = Matrix(qr(ψk).Q)
        ψ0 = [ortho(randn(Complex{T}, length(G_vectors(kpt)), N))
              for kpt in basis.kpoints]
    else
        for ik in 1:Nk
            ψ0[ik] = ψ0[ik][:,1:N]
        end
    end
    ## keep only occupied orbitals
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
                            krylov_verb=verb)

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

    energies, H = energy_hamiltonian(basis, φ, occupation; ρ=ρ)

    # Rayleigh-Ritz
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
    (ham=H, basis=basis, energies=energies, converged=true,
     ρ=ρ, ψ=φ, eigenvalues=eigenvalues, occupation=occupation, εF=εF)
end
