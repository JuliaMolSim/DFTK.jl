"""
Obtain a guess density form diagonalising the core Hamiltonian associated to
`ham`, i.e. strip the non-linear parts. `compute_occupation` computes the
occupation values, taking into account potential smearing. `lobpcg_prec` is a
preconditioner for LOBPCG and `n_bands` the number of bands to compute.
"""
function guess_hcore(ham::Hamiltonian, n_bands, compute_occupation;
                     lobpcg_prec=PreconditionerKinetic(ham, α=0.1))
    hcore = Hamiltonian(ham.basis, pot_local=ham.pot_local,
                        pot_nonlocal=ham.pot_nonlocal)
    res = lobpcg(hcore, n_bands, prec=lobpcg_prec)
    @assert res.converged
    occupation = compute_occupation(ham.basis, res.λ, res.X)
    compute_density(ham.basis, res.X, occupation)
end
