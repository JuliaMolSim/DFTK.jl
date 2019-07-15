using NLsolve

"""
TODO docme
"""
function scf_nlsolve(ham::Hamiltonian, n_bands, compute_occupation, ρ;
                     tol=1e-6, lobpcg_prec=PreconditionerKinetic(ham, α=0.1),
                     max_iter=100)
    pw = ham.basis
    values_hartree = empty_potential(ham.pot_hartree)
    values_xc = empty_potential(ham.pot_xc)
    Psi = Array{Any}([nothing])  # TODO Find out precise type

    function foldρ(ρ)
        # Fold a complex array representing the Fourier transform of a purely real
        # quantity into a real array
        half = Int((length(ρ) + 1) / 2)
        ρcpx =  ρ[1:half]
        vcat(real(ρcpx), imag(ρcpx))
    end
    function unfoldρ(ρ)
        # Undo "foldρ"
        half = Int(length(ρ) / 2)
        ρcpx = ρ[1:half] + im * ρ[half+1:end]
        vcat(ρcpx, conj(reverse(ρcpx)[2:end]))
    end
    function compute_residual!(residual, ρ_folded)
        ρ = unfoldρ(ρ_folded)
        values_hartree = compute_potential!(values_hartree, ham.pot_hartree, ρ)
        values_xc = compute_potential!(values_xc, ham.pot_xc, ρ)
        res = lobpcg(ham, n_bands, pot_hartree_values=values_hartree,
                     pot_xc_values=values_xc, guess=Psi[1],
                     prec=lobpcg_prec, tol=tol / 100)
        Psi[1] = res.X
        occupation = compute_occupation(ham.basis, res.λ, res.X)
        ρ_new = compute_density(pw, res.X, occupation, tolerance_orthonormality=tol)

        residual .= foldρ(ρ_new) - ρ_folded
    end
    nlres = nlsolve(compute_residual!, foldρ(ρ), method=:anderson, m=5, xtol=tol,
                    ftol=0.0, show_trace=true)

    # Final LOBPCG to get eigenvalues and eigenvectors
    ρ = unfoldρ(nlres.zero)
    values_hartree = compute_potential!(values_hartree, ham.pot_hartree, ρ)
    values_xc = compute_potential!(values_xc, ham.pot_xc, ρ)
    res = lobpcg(ham, n_bands, pot_hartree_values=values_hartree,
                 pot_xc_values=values_xc, guess=Psi[1],
                 prec=lobpcg_prec, tol=tol / 100)
    occupation = compute_occupation(ham.basis, res.λ, res.X)

    (ρ=ρ, Psi=res.X, energies=res.λ, occupation=occupation,converged=converged(nlres),
     pot_hartree_values=values_hartree, pot_xc_values=values_xc)
end

