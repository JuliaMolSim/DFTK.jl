"""
Interpolate some data from one k-Point to another. The interpolation is fast, but not
necessarily exact or even normalised. Intended only to construct guesses for iterative
solvers
"""
function interpolate_at_kpoint(kpt_old, kpt_new, data_oldk::AbstractVecOrMat)
    if kpt_old == kpt_new
        return data_oldk
    end
    @assert length(kpt_old.basis) == size(data_oldk, 1)
    n_bands = size(data_oldk, 2)

    data_newk = similar(data_oldk, length(kpt_new.basis), n_bands) .= 0
    for (iold, inew) in enumerate(indexin(kpt_old.basis, kpt_new.basis))
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
                                 kpoints=ham.basis.kpoints, guess=nothing,
                                 prec_type=PreconditionerTPA, interpolate_kpoints=true,
                                 tol=1e-6, maxiter=200, n_conv_check=nothing)
    T = eltype(ham)
    results = Vector{Any}(undef, length(kpoints))

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
            # TODO The double conversion is needed due to an issue in Julia
            #      see https://github.com/JuliaLang/julia/pull/32979
            qrres = qr(randn(real(T), length(kpoints[ik].basis), nev_per_kpoint))
            guessk = Matrix{T}(qrres.Q)
        end
        @assert size(guessk) == (length(kpoints[ik].basis), nev_per_kpoint)

        prec = nothing
        prec_type !== nothing && (prec = prec_type(ham, kpt))
        results[ik] = eigensolver(kblock(ham, kpt), guessk;
                                  prec=prec, tol=tol, maxiter=maxiter,
                                  n_conv_check=n_conv_check)
    end

    # Transform results into a nicer datastructure
    (λ=[real.(res.λ) for res in results],
     X=[res.X for res in results],
     residual_norms=[res.residual_norms for res in results],
     iterations=[res.iterations for res in results],
     converged=all(res.converged for res in results),
     kpoints=kpoints
    )
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
include("diag_lobpcg_scipy.jl")
include("diag_lobpcg_hyper.jl")
include("diag_lobpcg_itsolve.jl")
include("diag_full.jl")
