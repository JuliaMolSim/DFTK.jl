using NLsolve

"""
This is a very simple and first version of an SCF function, which will definitely
need some polishing and some generalisation later on.
"""
function self_consistent_field(ham::Hamiltonian, n_bands::Int, n_filled::Int;
                               PsiGuess=nothing, tol=1e-6,
                               lobpcg_preconditioner=PreconditionerKinetic(ham, α=0.1))
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
        hcore = Hamiltonian(pot_local=ham.pot_local, pot_nonlocal=ham.pot_nonlocal)
        res = lobpcg(hcore, n_bands, preconditioner=lobpcg_preconditioner)
        @assert res.converged
        Psi = [Xsk[:, 1:end] for Xsk in res.X]
    else
        Psi = PsiGuess
    end
    @assert size(Psi[1], 2) == n_bands

    # Compute starting density
    ρ_Y = compute_density(pw, Psi, occupation, tolerance_orthonormality=10 * tol)

    # Precompute first objects for faster application of Hartree and XC terms
    # TODO This looks a little weird
    precomp_hartree = empty_precompute(ham.pot_hartree)
    precomp_xc = empty_precompute(ham.pot_xc)

    # TODO nlsolve does not work right now
    # # compute_residual closure for nlsolve.
    # # nlsolve cannot do complex ... unfortunately
    # #    TODO not actually a problem, since we can do a "toReal" Fourier-transform
    # function compute_residual!(residual, ρ_Y_folded)
    #     n_G = length(pw.Gs)
    #     @assert size(ρ_Y_folded) == (2 * n_G, )
    #     ρ_Y = ρ_Y[1:n_G] + im * ρ_Y_folded[n_G + 1:end]

    #     if precomp_hartree !== nothing
    #         orig = copy(precomp_hartree.values_Yst)
    #         precomp_hartree = precompute!(precomp_hartree, ham.pot_hartree, ρ_Y)
    #         precomp_xc = precompute!(precomp_xc, ham.pot_xc, ρ_Y)
    #         println("Precomp distance: $(maximum(abs.(precomp_hartree.values_Yst - orig)))  norm  $(maximum(abs.(precomp_hartree.values_Yst)))")
    #     end

    #     tol_lobpcg = tol / 100
    #     res = lobpcg(ham, n_bands, precomp_hartree=precomp_hartree,
    #                   precomp_xc=precomp_xc, guess=Psi,
    #                   preconditioner=lobpcg_preconditioner, tol=tol_lobpcg)
    #     for (i, λ) in enumerate(res.λ)
    #         println("λs $i:  ", λ)
    #     end
    #     Psi[:] = res.X

    #     ρ_Y_new = compute_density(pw, Psi, occupation,
    #                               tolerance_orthonormality=tol)

    #     residual .= [real(ρ_Y_new); imag(ρ_Y_new)] - ρ_Y_folded
    #     residual
    # end

    # ρ_Y_folded = [real(ρ_Y); imag(ρ_Y)]
    # res = nlsolve(compute_residual!, ρ_Y_folded, method=:anderson, m=5, xtol=tol,
    #               ftol=0.0, show_trace=true)
    # @assert converged(res)
    #
    # n_G = length(pw.Gs)
    # @assert maximum(abs.(res.zero[n_G + 1:end])) < tol
    # ρ_Y = res.zero[1:n_G]

    for i in 1:100
        if precomp_hartree !== nothing
            orig = copy(precomp_hartree.values_Yst)
            precomp_hartree = precompute!(precomp_hartree, ham.pot_hartree, ρ_Y)
            precomp_xc = precompute!(precomp_xc, ham.pot_xc, ρ_Y)
        end

        tol_lobpcg = tol / 100
        res = lobpcg(ham, n_bands, precomp_hartree=precomp_hartree,
                      precomp_xc=precomp_xc, guess=nothing, # =Psi
                      preconditioner=lobpcg_preconditioner, tol=tol_lobpcg)
        @assert res.converged
        Psi[:] = res.X

        ρ_Y_new = compute_density(pw, Psi, occupation,
                                  tolerance_orthonormality=tol)

        ndiff = norm(ρ_Y_new - ρ_Y)
        @printf "%4d %15.8g\n" i ndiff
        if 20 * ndiff < tol
            break
        end
        ρ_Y = ρ_Y_new * 0.2 + 0.8 * ρ_Y
    end

    precompute!(precomp_hartree, ham.pot_hartree, ρ_Y)
    precompute!(precomp_xc, ham.pot_xc, ρ_Y)
    return ρ_Y, precomp_hartree, precomp_xc
end
