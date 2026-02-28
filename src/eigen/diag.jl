import LinearAlgebra: ldiv!

# Wrapper to apply the scalar preconditioner to both components of a spinor
struct NoncollinearPreconditioner
    prec
    n_Gk::Int
end
function LinearAlgebra.ldiv!(Y, P::NoncollinearPreconditioner, X)
    ldiv!(view(Y, 1:P.n_Gk, :), P.prec, view(X, 1:P.n_Gk, :))
    ldiv!(view(Y, P.n_Gk+1:size(X, 1), :), P.prec, view(X, P.n_Gk+1:size(X, 1), :))
    Y
end
function LinearAlgebra.ldiv!(P::NoncollinearPreconditioner, X)
    ldiv!(P.prec, view(X, 1:P.n_Gk, :))
    ldiv!(P.prec, view(X, P.n_Gk+1:size(X, 1), :))
    X
end

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
    is_full = ham.basis.model.spin_polarization == :full

    for (ik, kpt) in enumerate(kpoints)
        n_Gk = length(G_vectors(ham.basis, kpt))
        n_dim = is_full ? 2 * n_Gk : n_Gk
        
        if n_Gk < nev_per_kpoint
            error("The size of the plane wave basis is $n_Gk, and you are asking for " *
                  "$nev_per_kpoint eigenvalues. Increase Ecut.")
        end
        
        # Get ψguessk
        if !isnothing(ψguess)
            if n_dim != size(ψguess[ik], 1)
                error("Mismatch in dimension between guess ($(size(ψguess[ik], 1))) and " *
                      "Hamiltonian ($n_dim)")
            end
            nev_guess = size(ψguess[ik], 2)
            if nev_guess > nev_per_kpoint
                ψguessk = ψguess[ik][:, 1:nev_per_kpoint]
            elseif nev_guess == nev_per_kpoint
                ψguessk = ψguess[ik]
            else
                X0 = similar(ψguess[ik], n_dim, nev_per_kpoint)
                X0[:, 1:nev_guess] = ψguess[ik]
                X0[:, nev_guess+1:end] = randn(eltype(X0), n_dim, nev_per_kpoint - nev_guess)
                ψguessk = ortho_qr(X0)
            end
        elseif interpolate_kpoints && ik > 1
            # use information from previous k-point
            if is_full
                # Interpolate the up and down spin components separately
                n_Gk_prev = length(G_vectors(ham.basis, kpoints[ik - 1]))
                ψ_up = interpolate_kpoint(results[ik - 1].X[1:n_Gk_prev, :], ham.basis, kpoints[ik - 1], ham.basis, kpoints[ik])
                ψ_dn = interpolate_kpoint(results[ik - 1].X[n_Gk_prev+1:end, :], ham.basis, kpoints[ik - 1], ham.basis, kpoints[ik])
                ψguessk = vcat(ψ_up, ψ_dn)
            else
                ψguessk = interpolate_kpoint(results[ik - 1].X, ham.basis, kpoints[ik - 1],
                                             ham.basis, kpoints[ik])
            end
        else
            ψguessk = random_orbitals(ham.basis, kpt, nev_per_kpoint)
        end
        @assert size(ψguessk) == (n_dim, nev_per_kpoint)

        prec = nothing
        if !isnothing(prec_type)
            prec_base = prec_type(ham[ik])
            prec = is_full ? NoncollinearPreconditioner(prec_base, n_Gk) : prec_base
        end
        
        results[ik] = eigensolver(ham[ik], ψguessk;
                                  prec, tol, miniter, maxiter, n_conv_check)
    end

    # Transform results into a nicer datastructure
    # TODO It feels inconsistent to put λ onto the CPU here but none of the other objects.
    #      Better have this handled by the caller of diagonalize_all_kblocks.
    #      Note further that lobpcg_hyper by default puts the eigenvalues
    #      on the CPU ... even if the next line is removed.
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