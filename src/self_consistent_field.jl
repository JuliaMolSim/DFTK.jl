using NLsolve

"""
This is a very simple and first version of an SCF function, which will definitely
need some polishing and some generalisation later on.
"""
function self_consistent_field(ham::Hamiltonian, n_bands::Int, n_filled::Int;
                               PsiGuess=nothing, tol=1e-6)
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
        res = lobpcg(hcore, n_bands + 1, preconditioner=PreconditionerKinetic(ham, α=0.1))
        @assert res.converged
        Psi = [Xsk[:, 2:end] for Xsk in res.X]
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

    # compute_residual closure for nlsolve.
    # nlsolve cannot do complex ... unfortunately
    #    TODO not actually a problem, since we can do a "toReal" Fourier-transform
    function compute_residual!(residual, ρ_Y_folded)
        n_G = length(pw.Gs)
        @assert size(ρ_Y_folded) == (2 * n_G, )
        ρ_Y = ρ_Y[1:n_G] + im * ρ_Y_folded[n_G + 1:end]
        precompute!(precomp_hartree, ham.pot_hartree, ρ_Y)
        precompute!(precomp_xc, ham.pot_xc, ρ_Y)

        tol_lobpcg = tol / 100
        precond = PreconditionerKinetic(ham, α=0.1)
        res = lobpcg(ham, n_bands, precomp_hartree=precomp_hartree,
                     precomp_xc=precomp_xc, guess=Psi, preconditioner=precond,
                     tol=tol_lobpcg)
        Psi[:] = res.X

        ρ_Y_new = compute_density(pw, Psi, occupation,
                                  tolerance_orthonormality=tol)
        residual .= [real(ρ_Y_new); imag(ρ_Y_new)] - ρ_Y_folded
    end

    ρ_Y_folded = [real(ρ_Y); imag(ρ_Y)]
    res = nlsolve(compute_residual!, ρ_Y_folded, method=:anderson, m=5, xtol=tol,
                  ftol=0.0, show_trace=true)
    @assert converged(res)

    n_G = length(pw.Gs)
    @assert maximum(abs.(res.zero[n_G + 1:end])) < 1e-12
    ρ_Y = res.zero[1:n_G]
    precompute!(precomp_hartree, ham.pot_hartree, ρ_Y)
    precompute!(precomp_xc, ham.pot_xc, ρ_Y)
    return ρ_Y, precomp_hartree, precomp_xc
end
