"""
Docme
"""
function self_consistent_field(ham::Hamiltonian, n_bands::Int, n_filled::Int;
                               ρ=nothing, tol=1e-6,
                               lobpcg_prec=PreconditionerKinetic(ham, α=0.1),
                               max_iter=100, algorithm=:scf_nlsolve)
    function compute_occupation(basis, energies, Psi)
        occupation_zero_temperature(basis, energies, Psi, 2 * n_filled)
    end
    if ρ === nothing
        ρ = guess_hcore(ham, n_bands, compute_occupation, lobpcg_prec=lobpcg_prec)
    end
    if algorithm == :scf_nlsolve
        res = scf_nlsolve(ham, n_bands, compute_occupation, ρ, tol=tol,
                          lobpcg_prec=lobpcg_prec, max_iter=max_iter)
    elseif algorithm == :scf_damped
        res = scf_damped(ham, n_bands, compute_occupation, ρ, tol=tol,
                         lobpcg_prec=lobpcg_prec, max_iter=max_iter)
    else
        error("Unknown algorithm " * str(algorithm))
    end

    res.ρ, res.pot_hartree_values, res.pot_xc_values
end
