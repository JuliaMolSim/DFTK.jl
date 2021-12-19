using ProgressMeter

@doc raw"""
Function for diagonalising each ``k``-Point blow of ham one step at a time.
Some logic for interpolating between ``k``-points is used if `interpolate_kpoints`
is true and if no guesses are given. `eigensolver` is the iterative eigensolver
that really does the work, operating on a single ``k``-Block.
`eigensolver` should support the API `eigensolver(A, X0; prec, tol, maxiter)`
`prec_type` should be a function that returns a preconditioner when called as `prec(ham, kpt)`
"""
function diagonalize_all_kblocks(eigensolver, ham::Hamiltonian, nev_per_kpoint::Int;
                                 guess=nothing,
                                 prec_type=PreconditionerTPA, interpolate_kpoints=true,
                                 tol=1e-6, miniter=1, maxiter=200, n_conv_check=nothing,
                                 show_progress=false)
    T = complex(eltype(ham.basis))
    kpoints = ham.basis.kpoints
    results = Vector{Any}(undef, length(kpoints))

    progress = nothing
    if show_progress
        progress = Progress(length(kpoints), desc="Diagonalising Hamiltonian kblocks: ")
    end
    for (ik, kpt) in enumerate(kpoints)
        n_Gk = length(G_vectors(ham.basis, kpoints[ik]))
        if n_Gk < nev_per_kpoint
            error("The size of the plane wave basis is $n_Gk, and you are asking for " *
                  "$nev_per_kpoint eigenvalues. Increase Ecut.")
        end
        # Get guessk
        @timing "QR orthonormalization" begin
            if guess !== nothing
                # guess provided
                guessk = guess[ik]
            elseif interpolate_kpoints && ik > 1
                # use information from previous k-point
                X0 = interpolate_kpoint(results[ik - 1].X, ham.basis, kpoints[ik - 1],
                                        ham.basis, kpoints[ik])
                guessk = ortho_qr(X0)  # Re-orthogonalize and renormalize
            else
                # random initial guess
                guessk = ortho_qr(randn(T, n_Gk, nev_per_kpoint))
            end
        end
        @assert size(guessk) == (n_Gk, nev_per_kpoint)

        prec = nothing
        prec_type !== nothing && (prec = prec_type(ham.basis, kpt))
        results[ik] = eigensolver(ham.blocks[ik], guessk;
                                  prec=prec, tol=tol, miniter=miniter, maxiter=maxiter,
                                  n_conv_check=n_conv_check)

        # Update progress bar if desired
        !isnothing(progress) && next!(progress)
    end

    # Transform results into a nicer datastructure
    (λ=[real.(res.λ) for res in results],
     X=[res.X for res in results],
     residual_norms=[res.residual_norms for res in results],
     iterations=[res.iterations for res in results],
     converged=all(res.converged for res in results),
     n_matvec=sum(res.n_matvec for res in results))
end

@doc raw"""
Function to select a subset of eigenpairs on each ``k``-Point. Works on the
Tuple returned by `diagonalize_all_kblocks`.
"""
function select_eigenpairs_all_kblocks(eigres, range)
    merge(eigres, (λ=[λk[range] for λk in eigres.λ],
                   X=[Xk[:, range] for Xk in eigres.X],
                   residual_norms=[resk[range] for resk in eigres.residual_norms]))
end

# The actual implementations using the above primitives
include("diag_full.jl")
include("diag_lobpcg_hyper.jl")
