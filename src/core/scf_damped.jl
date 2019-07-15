"""
TODO docme
"""
function scf_damped(ham::Hamiltonian, n_bands, compute_occupation, ρ;
                    tol=1e-6, lobpcg_prec=PreconditionerKinetic(ham, α=0.1),
                    max_iter=100, damping=0.20)
    energies = nothing
    occupation = nothing
    Psi=nothing
    converged = false
    values_hartree = empty_potential(ham.pot_hartree)
    values_xc = empty_potential(ham.pot_xc)
    for i in 1:max_iter
        values_hartree = compute_potential!(values_hartree, ham.pot_hartree, ρ)
        values_xc = compute_potential!(values_xc, ham.pot_xc, ρ)

        res = lobpcg(ham, n_bands, pot_hartree_values=values_hartree,
                      pot_xc_values=values_xc, guess=Psi,
                      prec=lobpcg_prec, tol=tol / 100)
        @assert res.converged
        Psi = res.X
        energies = res.λ

        occupation = compute_occupation(ham.basis, res.λ, Psi)
        ρ_new = compute_density(ham.basis, Psi, occupation, tolerance_orthonormality=tol)

        # TODO Print statements should not be here
        ndiff = norm(ρ_new - ρ)
        @printf "%4d %18.8g %s  " i ndiff res.implementation
        println(res.iterations)
        if 20 * ndiff < tol
            converged = true
            break
        end

        ρ = ρ_new * damping + (1 - damping) * ρ
    end

    (ρ=ρ, Psi=Psi, energies=energies, occupation=occupation,
     pot_hartree_values=compute_potential!(values_hartree, ham.pot_hartree, ρ),
     pot_xc_values=compute_potential!(values_xc, ham.pot_xc, ρ),
     converged=converged)
end

