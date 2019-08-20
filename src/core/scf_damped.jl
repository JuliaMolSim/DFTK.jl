"""
TODO docme
"""
function scf_damped(ham::Hamiltonian, n_bands, compute_occupation, ρ;
                    tol=1e-6, lobpcg_prec=PreconditionerKinetic(ham, α=0.1),
                    max_iter=100, damping=0.20)
    pw = ham.basis
    T = real(eltype(ρ))

    orben = nothing
    converged = false
    values_hartree = empty_potential(ham.pot_hartree)
    values_xc = empty_potential(ham.pot_xc)
    energies = Dict{Symbol, real(eltype(ρ))}()

    # Initialise guess for wave function and occupation
    Psi = [Matrix(qr(randn(Complex{T}, length(pw.basis_wf[ik]), n_bands)).Q)
           for ik in 1:length(pw.kpoints)]
    occupation = [zeros(n_bands) for ik in 1:length(pw.kpoints)]

    for i in 1:max_iter
        # TODO The following three lines of code occur quite a few times
        #      ... perhaps one should make them an extra function
        update_energies_1e!(energies, ham, ρ, Psi, occupation)
        update_energies_potential!(energies, values_hartree, ham.pot_hartree, ρ)
        update_energies_potential!(energies, values_xc, ham.pot_xc, ρ)

        res = lobpcg(ham, n_bands, pot_hartree_values=values_hartree,
                      pot_xc_values=values_xc, guess=Psi,
                      prec=lobpcg_prec, tol=tol / 10)
        @assert res.converged
        Psi = res.X
        orben = res.λ

        occupation = compute_occupation(ham.basis, res.λ, Psi)
        ρ_new = compute_density(ham.basis, Psi, occupation)

        # TODO Print statements should not be here
        ndiff = norm(ρ_new - ρ)
        @printf "%4d %18.8g %s  " i ndiff res.implementation
        println(res.iterations)

        if 20 * ndiff < tol
            ρ = ρ_new
            converged = true
            break
        end

        ρ = ρ_new * damping + (1 - damping) * ρ
    end

    update_energies_1e!(energies, ham, ρ, Psi, occupation)
    update_energies_potential!(energies, values_hartree, ham.pot_hartree, ρ)
    update_energies_potential!(energies, values_xc, ham.pot_xc, ρ)
    (ρ=ρ, Psi=Psi, orben=orben, occupation=occupation, energies=energies,
     pot_hartree_values=values_hartree, pot_xc_values=values_xc, converged=converged)
end

