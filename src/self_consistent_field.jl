using NLsolve

"""
This is a very simple and first version of an SCF function, which will definitely
need some polishing and some generalisation later on.
"""
function self_consistent_field(ham::Hamiltonian, n_bands::Int, n_filled::Int;
                               PsiGuess=nothing, tol=1e-6,
                               lobpcg_prec=PreconditionerKinetic(ham, α=0.1))
    pw = ham.basis
    n_k = length(pw.kpoints)

    occupation = Vector{Vector{Float64}}(undef, n_k)
    for ik in 1:n_k
        occupation[ik] = zero(1:n_bands)
        occupation[ik][1:n_filled] .= 2
    end

    # This guess-forming stuff should probably be done somewhere outside this code
    Psi = nothing
    if PsiGuess === nothing
        hcore = Hamiltonian(ham.basis, pot_local=ham.pot_local,
                            pot_nonlocal=ham.pot_nonlocal)
        res = lobpcg(hcore, n_bands, prec=lobpcg_prec)
        @assert res.converged
        Psi = [Xsk[:, 1:end] for Xsk in res.X]
    else
        Psi = PsiGuess
    end
    @assert size(Psi[1], 2) == n_bands

    # Compute starting density
    ρ_Y = compute_density(pw, Psi, occupation, tolerance_orthonormality=20 * tol)

    # Precompute first objects for faster application of Hartree and XC terms
    # TODO This looks a little weird
    precomp_hartree = empty_potential(ham.pot_hartree)
    precomp_xc = empty_potential(ham.pot_xc)

    use_nlsolve = false
    if use_nlsolve
        # compute_residual closure for nlsolve.
        # nlsolve cannot do complex ... unfortunately
        #    TODO not actually a problem, since we can do a "toReal" Fourier-transform
        function compute_residual!(residual, ρ_Y_folded)
            n_G = prod(pw.grid_size)
            @assert size(ρ_Y_folded) == (2 * n_G, )
            ρ_Y = ρ_Y[1:n_G] + im * ρ_Y_folded[n_G + 1:end]

            tol_lobpcg = tol / 100
            precomp_hartree = compute_potential!(precomp_hartree, ham.pot_hartree, ρ_Y)
            precomp_xc = compute_potential!(precomp_xc, ham.pot_xc, ρ_Y)
            res = lobpcg(ham, n_bands, precomp_hartree=precomp_hartree,
                          precomp_xc=precomp_xc, guess=Psi,
                          prec=lobpcg_prec, tol=tol_lobpcg)
            for (i, λ) in enumerate(res.λ)
                println("λs $i:  ", λ)
            end
            Psi[:] = res.X

            ρ_Y_new = compute_density(pw, Psi, occupation,
                                      tolerance_orthonormality=tol)

            residual .= [real(ρ_Y_new); imag(ρ_Y_new)] - ρ_Y_folded
            residual
        end

        ρ_Y_folded = [real(ρ_Y); imag(ρ_Y)]
        res = nlsolve(compute_residual!, ρ_Y_folded, method=:anderson, m=5, xtol=tol,
                      ftol=0.0, show_trace=true)
        @assert res.converged

        n_G = prod(pw.grid_size)
        @assert maximum(abs.(res.zero[n_G + 1:end])) < tol
        ρ_Y = res.zero[1:n_G]
    else
        for i in 1:100
            if precomp_hartree !== nothing
                precomp_hartree = compute_potential!(precomp_hartree, ham.pot_hartree, ρ_Y)
                precomp_xc = compute_potential!(precomp_xc, ham.pot_xc, ρ_Y)
            end

            tol_lobpcg = tol / 100
            if i == 1
                tol_lobpcg = tol * 10
            end
            res = lobpcg(ham, n_bands, pot_hartree_values=precomp_hartree,
                          pot_xc_values=precomp_xc, guess=Psi,
                          prec=lobpcg_prec, tol=tol_lobpcg)
            @assert res.converged
            Psi[:] = res.X

            ρ_Y_new = compute_density(pw, Psi, occupation,
                                      tolerance_orthonormality=tol)

            ndiff = norm(ρ_Y_new - ρ_Y)
            @printf "%4d %18.8g %s  " i ndiff res.implementation
            println(res.iterations)

            if 20 * ndiff < tol
                for (iλ, λ) in enumerate(res.λ)
                    println("    $iλ $λ")
                end
                break
            end
            damp = 0.25
            ρ_Y = ρ_Y_new * damp + (1 - damp) * ρ_Y
        end
    end # use_nlsolve

    compute_potential!(precomp_hartree, ham.pot_hartree, ρ_Y)
    compute_potential!(precomp_xc, ham.pot_xc, ρ_Y)
    return ρ_Y, precomp_hartree, precomp_xc
end
