# This file contains functions to perform newton step and a newton solver for
# DFTK problems.


# we compute the residual associated to a set of planewave φ, that is to say
# H(φ)*φ - λ.*φ where λ is the set of rayleigh coefficients associated to the
# φ
# we also return the egval set for further computations
function compute_residual(basis::PlaneWaveBasis{T}, φ, occ) where T

    # necessary quantities
    Nk = length(basis.kpoints)
    ρ = compute_density(basis, φ, occ)
    energies, H = energy_hamiltonian(basis, φ, occ; ρ=ρ[1])

    # compute residual
    res = similar(φ)
    for ik = 1:Nk
        φk = φ[ik]
        N = size(φk, 2)
        Hk = H.blocks[ik]
        # eigenvalues as rayleigh coefficients
        egvalk = [φk[:,i]'*(Hk*φk[:,i]) for i = 1:N]
        # compute residual at given kpoint as H(φ)φ - λφ
        rk = Hk*φk - hcat([egvalk[i] * φk[:,i] for i = 1:N]...)
        res[ik] = rk
    end

    return res
end


# perform a newton step : we take as given a planewave set φ and we return the
# newton step φ - δφ (after proper orthonormalization) where δφ solves Jac * δφ = res
function newton_step(basis::PlaneWaveBasis{T}, φ, res, occ;
                     φproj=nothing, tol_krylov=1e-12) where T

    # necessary quantities
    N = size(φ[1],2)
    Nk = length(basis.kpoints)
    ortho(ψk) = Matrix(qr(ψk).Q)

    # if no φ is specified to define the projection onto the tangent plane, we
    # use the current φ
    if φproj === nothing
        φproj = φ
    end
    ρproj = compute_density(basis, φproj, occ)
    energies, Hproj = energy_hamiltonian(basis, φproj, occ; ρ=ρproj[1])
    egvalproj = [ zeros(Complex{T}, size(occ[ik])) for ik = 1:Nk ]
    for ik = 1:Nk
        Hk = Hproj.blocks[ik]
        φk = φproj[ik]
        egvalproj[ik] = [φk[:,i]'*(Hk*φk[:,i]) for i = 1:N]
    end

    # packing routines
    pack, unpack, packed_proj = packing(basis, φproj)

    # solve linear system with KrlyovKit
    function f(x)
        δφ = unpack(x)
        ΩpKx = ΩplusK(basis, δφ, φproj, ρproj[1], Hproj, egvalproj, occ)
        pack(ΩpKx)
    end

    # project res on the good tangent space
    res = proj(res, φproj)

    δφ, info = linsolve(f, pack(res);
                        tol=tol_krylov, verbosity=1,
                        orth=OrthogonalizeAndProject(packed_proj, pack(φproj)))
    δφ = unpack(δφ)
    δφ = proj(δφ, φ)
    φ_newton = similar(φ)

    for ik = 1:Nk
        φk = φ[ik]
        δφk = δφ[ik]
        N = size(φk,2)
        for i = 1:N
            φk[:,i] = φk[:,i] - δφk[:,i]
        end
        φk = ortho(φk)
        φ_newton[ik] = φk
    end
    (φ_newton, δφ)
end

# newton algorithm
function newton(basis::PlaneWaveBasis{T}; ψ0=nothing,
                tol=1e-6, max_iter=100, φproj=nothing) where T

    ## setting parameters
    model = basis.model
    @assert model.spin_polarization in (:none, :spinless)
    @assert model.temperature == 0 # temperature is not yet supported
    filled_occ = DFTK.filled_occupation(model)
    N = div(model.n_electrons, filled_occ)

    ## number of kpoints
    Nk = length(basis.kpoints)
    occupation = [filled_occ * ones(T, N) for ik = 1:Nk]

    ## starting point and orthonormalization routine
    if ψ0 === nothing
        ortho(ψk) = Matrix(qr(ψk).Q)
        ψ0 = [ortho(randn(Complex{T}, length(G_vectors(kpt)), N))
              for kpt in basis.kpoints]
    end

    ## error list for convergence plots
    err_list = []
    err_ref_list = []
    k_list = []

    err = 1
    k = 0

    # orbitals to be updated along the iterations
    φ = deepcopy(ψ0)

    while err > tol && k < max_iter
        k += 1
        println("Iteration $(k)...")
        append!(k_list, k)

        if φproj === nothing
            ϕ = φ
        else
            ϕ = φproj
        end

        # compute next step
        ρ = compute_density(basis, φ, occupation)
        res = compute_residual(basis, φ, occupation)
        φ, δφ = newton_step(basis, φ, res, occupation; φproj=ϕ)

        # compute error on the norm
        ρ_next = compute_density(basis, φ, occupation)
        err = norm(ρ_next[1].real - ρ[1].real)
        append!(err_list, err)
        append!(err_ref_list, norm(ρ_next[1].real - scfres.ρ.real))
    end

    # plot results
    figure()
    title("Convergence of newton algorithm")
    semilogy(k_list, err_list, "x-", label="|ρ^{k+1} - ρ^k|")
    semilogy(k_list, err_ref_list, "x-", label="|ρ^k - ρref|")
    xlabel("iterations")
    legend()
end


