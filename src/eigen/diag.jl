@doc raw"""
Function for diagonalising each ``k``-Point blow of ham one step at a time.
Some logic for interpolating between ``k``-points is used if `interpolate_kpoints`
is true and if no guesses are given. `eigensolver` is the iterative eigensolver
that really does the work, operating on a single ``k``-Block.
`eigensolver` should support the API `eigensolver(A, X0; prec, tol, maxiter)`
`prec_type` should be a function that returns a preconditioner when called as `prec(ham, kpt)`
"""
function diagonalize_all_kblocks(eigensolver, ham::Hamiltonian, nev_per_kpoint::Int;
                                 ψguess=nothing,
                                 prec_type=PreconditionerTPA, interpolate_kpoints=true,
                                 tol=1e-6, miniter=1, maxiter=100, n_conv_check=nothing)
    kpoints = ham.basis.kpoints
    results = Vector{Any}(undef, length(kpoints))

    for (ik, kpt) in enumerate(kpoints)
        n_Gk = length(G_vectors(ham.basis, kpt))
        if n_Gk < nev_per_kpoint
            error("The size of the plane wave basis is $n_Gk, and you are asking for " *
                  "$nev_per_kpoint eigenvalues. Increase Ecut.")
        end
        # Get ψguessk
        if !isnothing(ψguess)
            if n_Gk != size(ψguess[ik], 1)
                error("Mismatch in dimension between guess ($(size(ψguess[ik], 1)) and " *
                      "Hamiltonian ($n_Gk)")
            end
            nev_guess = size(ψguess[ik], 2)
            if nev_guess > nev_per_kpoint
                ψguessk = ψguess[ik][:, 1:nev_per_kpoint]
            elseif nev_guess == nev_per_kpoint
                ψguessk = ψguess[ik]
            else
                X0 = similar(ψguess[ik], n_Gk, nev_per_kpoint)
                X0[:, 1:nev_guess] = ψguess[ik]
                X0[:, nev_guess+1:end] = randn(eltype(X0), n_Gk, nev_per_kpoint - nev_guess)
                ψguessk = ortho_qr(X0)
            end
        elseif interpolate_kpoints && ik > 1
            # use information from previous k-point
            ψguessk = interpolate_kpoint(results[ik - 1].X, ham.basis, kpoints[ik - 1],
                                         ham.basis, kpoints[ik])
        else
            ψguessk = random_orbitals(ham.basis, kpt, nev_per_kpoint)
        end
        @assert size(ψguessk) == (n_Gk, nev_per_kpoint)

        prec = nothing
        !isnothing(prec_type) && (prec = prec_type(ham[ik]))
        results[ik] = eigensolver(ham[ik], ψguessk;
                                  prec, tol, miniter, maxiter, n_conv_check)
    end

    # Transform results into a nicer datastructure
    # TODO It feels inconsistent to put λ onto the CPU here but none of the other objects.
    #      Better have this handled by the caller of diagonalize_all_kblocks.
    (; λ=[to_cpu(real.(res.λ)) for res in results],  # Always get onto the CPU
     X=[res.X for res in results],
     residual_norms=[res.residual_norms for res in results],
     n_iter=[res.n_iter for res in results],
     converged=all(res.converged for res in results),
     n_matvec=sum(res.n_matvec for res in results))
end

@doc raw"""
Function to select a subset of eigenpairs on each ``k``-Point. Works on the
Tuple returned by `diagonalize_all_kblocks`.
"""
function select_eigenpairs_all_kblocks(eigres, range)
    merge(eigres, (; λ=[λk[range] for λk in eigres.λ],
                   X=[Xk[:, range] for Xk in eigres.X],
                   residual_norms=[resk[range] for resk in eigres.residual_norms]))
end

# The actual implementations using the above primitives
include("diag_full.jl")
include("diag_lobpcg_hyper.jl")
