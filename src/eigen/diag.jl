using ProgressMeter

"""
Interpolate some data from one k-Point to another. The interpolation is fast, but not
necessarily exact or even normalised. Intended only to construct guesses for iterative
solvers
"""
function interpolate_at_kpoint(kpt_old, kpt_new, data_oldk::AbstractVecOrMat)
    if kpt_old == kpt_new
        return data_oldk
    end
    @assert length(G_vectors(kpt_old)) == size(data_oldk, 1)
    n_bands = size(data_oldk, 2)

    data_newk = similar(data_oldk, length(G_vectors(kpt_new)), n_bands) .= 0
    for (iold, inew) in enumerate(indexin(G_vectors(kpt_old), G_vectors(kpt_new)))
        inew !== nothing && (data_newk[inew, :] = data_oldk[iold, :])
    end
    data_newk
end

@doc raw"""
Function for diagonalising each ``k``-Point blow of ham one step at a time.
Some logic for interpolating between ``k``-Points is used if `interpolate_kpoints`
is true and if no guesses are given. `eigensolver` is the iterative eigensolver
that really does the work, operating on a single ``k``-Block.
`eigensolver` should support the API `eigensolver(A, X0; prec, tol, maxiter)`
`prec_type` should be a function that returns a preconditioner when called as `prec(ham, kpt)`
"""
function diagonalise_all_kblocks(eigensolver, ham::Hamiltonian, nev_per_kpoint::Int;
                                 guess=nothing,
                                 prec_type=PreconditionerTPA, interpolate_kpoints=true,
                                 tol=1e-6, maxiter=200, n_conv_check=nothing,
                                 show_progress=false)
    T = complex(eltype(ham.basis))
    kpoints = ham.basis.kpoints
    results = Vector{Any}(undef, length(kpoints))

    progress = nothing
    if show_progress
        progress = Progress(length(kpoints), desc="Diagonalising Hamiltonian kblocks: ")
    end
    for (ik, kpt) in enumerate(kpoints)
        # Get guessk
        if guess != nothing
            # guess provided
            guessk = guess[ik]
        elseif interpolate_kpoints && ik > 1
            # use information from previous kpoint
            # TODO ensure better traversal of kpoints so that the interpolation is relevant
            X0 = interpolate_at_kpoint(kpoints[ik - 1], kpoints[ik], results[ik - 1].X)
            guessk = Matrix{T}(qr(X0).Q)
        else
            # random initial guess
            qrres = qr(randn(T, length(G_vectors(kpoints[ik])), nev_per_kpoint))
            guessk = Matrix{T}(qrres.Q)
        end
        @assert size(guessk) == (length(G_vectors(kpoints[ik])), nev_per_kpoint)

        prec = nothing
        prec_type !== nothing && (prec = prec_type(ham.basis, kpt))
        results[ik] = eigensolver(ham.blocks[ik], guessk;
                                  prec=prec, tol=tol, maxiter=maxiter,
                                  n_conv_check=n_conv_check)

        # Update progress bar if desired
        !isnothing(progress) && next!(progress)
    end

    # Transform results into a nicer datastructure
    (λ=[real.(res.λ) for res in results],
     X=[res.X for res in results],
     residual_norms=[res.residual_norms for res in results],
     iterations=[res.iterations for res in results],
     converged=all(res.converged for res in results))
end

@doc raw"""
Function to select a subset of eigenpairs on each ``k``-Point. Works on the
Tuple returned by `diagonalise_all_kblocks`.
"""
function select_eigenpairs_all_kblocks(eigres, range)
    merge(eigres, (λ=[λk[range] for λk in eigres.λ],
                   X=[Xk[:, range] for Xk in eigres.X],
                   residual_norms=[resk[range] for resk in eigres.residual_norms]))
end

# The actual implementations using the above primitives
include("diag_lobpcg_hyper.jl")
include("diag_lobpcg_itsolve.jl")
include("diag_full.jl")
