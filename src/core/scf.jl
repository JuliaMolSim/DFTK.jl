using NLsolve

# gets a new density ρ by diagonalizing the Hamiltonian. If Psi is
# provided, overwrite it with the new wavefunctions
function new_density(ham::Hamiltonian, n_bands, compute_occupation, ρ, lobpcg_tol;
                     lobpcg_prec=PreconditionerKinetic(ham, α=0.1), Psi=nothing)
    pw = ham.basis
    T = real(eltype(ρ))
    # Initialize guess wavefunctions if needed
    if Psi == nothing
        Psi = [Matrix(qr(randn(Complex{T}, length(pw.basis_wf[ik]), n_bands)).Q)
               for ik in 1:length(pw.kpoints)]
    end
    values_hartree = empty_potential(ham.pot_hartree)
    values_xc = empty_potential(ham.pot_xc)
    energies = Dict{Symbol, real(eltype(ρ))}()
    update_energies_potential!(energies, values_hartree, ham.pot_hartree, ρ)
    update_energies_potential!(energies, values_xc, ham.pot_xc, ρ)


    # Initialise guess for wave function and occupation
    Psi = [Matrix(qr(randn(Complex{T}, length(pw.basis_wf[ik]), n_bands)).Q)
           for ik in 1:length(pw.kpoints)]

    res = lobpcg(ham, n_bands, pot_hartree_values=values_hartree,
                 pot_xc_values=values_xc, guess=Psi,
                 prec=lobpcg_prec, tol=lobpcg_tol)
    @assert res.converged
    Psi .= res.X
    occupation = compute_occupation(ham.basis, res.λ, res.X)
    ρ_new = compute_density(pw, res.X, occupation,
                            tolerance_orthonormality=100lobpcg_tol)
end

# the fp_solver must accept being called like fp_solver(f, x0, tol, maxiter),
# where f(x) is the fixed-point map. It must return an object
# supporting res.sol and res.converged

function scf_nlsolve_solver(m)
    function fp_solver(f, x0, tol, max_iter)
        res = nlsolve(x -> f(x) - x, x0, method=:anderson, m=m, xtol=tol,
                      ftol=0.0, show_trace=true, iterations=max_iter)
        (sol=res.zero, converged=converged(res))
    end
    fp_solver
end
function scf_damping_solver(β)
    function fp_solver(f, x0, tol, max_iter)
        converged = false
        x = copy(x0)
        for i in 1:max_iter
            x_new = f(x)

            # TODO Print statements should not be here
            ndiff = norm(x_new - x)
            @printf "%4d %18.8g\n" i ndiff

            if 20 * ndiff < tol
                x = x_new
                converged = true
                break
            end

            x = @. β * x_new + (1 - β) * x
        end
        (sol=x, converged=converged)
    end
    fp_solver
end

function scf(ham::Hamiltonian, n_bands, compute_occupation, ρ, fp_solver;
             tol=1e-6, lobpcg_prec=PreconditionerKinetic(ham, α=0.1),
             max_iter=100, lobpcg_tol=tol / 100)
    pw = ham.basis
    T = real(eltype(ρ))
    Psi = [Matrix(qr(randn(Complex{T}, length(pw.basis_wf[ik]), n_bands)).Q)
           for ik in 1:length(pw.kpoints)]

    # TODO remove foldρ and unfoldρ when https://github.com/JuliaNLSolvers/NLsolve.jl/pull/217 is in a release
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

    fp_map(ρ) = foldρ(new_density(ham, n_bands, compute_occupation, unfoldρ(ρ), lobpcg_tol, lobpcg_prec=lobpcg_prec, Psi=Psi))

    nlres = fp_solver(fp_map, foldρ(ρ), tol, max_iter)
    ρ = unfoldρ(nlres.sol)
    energies = Dict{Symbol, real(eltype(ρ))}()
    values_hartree = empty_potential(ham.pot_hartree)
    values_xc = empty_potential(ham.pot_xc)
    update_energies_potential!(energies, values_hartree, ham.pot_hartree, ρ)
    update_energies_potential!(energies, values_xc, ham.pot_xc, ρ)

    # Final LOBPCG to get eigenvalues and eigenvectors
    res = lobpcg(ham, n_bands, pot_hartree_values=values_hartree,
                 pot_xc_values=values_xc, guess=Psi,
                 prec=lobpcg_prec, tol=lobpcg_tol)

    occupation = compute_occupation(ham.basis, res.λ, res.X)
    update_energies_1e!(energies, ham, ρ, res.X, occupation)

    (ρ=ρ, Psi=res.X, orben=res.λ, occupation=occupation, energies=energies,
     pot_hartree_values=values_hartree, pot_xc_values=values_xc, converged=nlres.converged)
end
