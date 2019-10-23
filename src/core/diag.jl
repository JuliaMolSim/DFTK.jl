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
is true and if no guesses are given. The `kernel` is the iterative eigensolver
that really does the work, operating on a single ``k``-Block.
"""
function diagonalise_all_kblocks(kernel, ham::Hamiltonian, nev_per_kpoint::Int;
                                 kpoints=ham.basis.kpoints, guess=nothing,
                                 prec=nothing, interpolate_kpoints=true, tol=1e-6,
                                 maxiter=200, kwargs...)
    T = eltype(ham)
    results = Vector{Any}(undef, length(kpoints))

    # Functions simplifying stuff to be done for each k-Point
    function get_guessk(guess, ik)
        # TODO Proper error messages
        @assert length(guess) ≥ length(kpoints)
        @assert size(guess[ik], 2) == nev_per_kpoint
        @assert size(guess[ik], 1) == length(kpoints[ik].basis)
        guess[ik]
    end
    function get_guessk(::Nothing, ik)
        if ik <= 1 || !interpolate_kpoints
            # TODO The double conversion is needed due to an issue in Julia
            #      see https://github.com/JuliaLang/julia/pull/32979
            qrres = qr(randn(real(T), length(kpoints[ik].basis), nev_per_kpoint))
            m = Matrix{T}(Matrix(qrres.Q))
        else  # Interpolate from previous k-Point
            X0 = interpolate_at_kpoint(kpoints[ik - 1], kpoints[ik], results[ik - 1].X)
            X0 = Matrix{T}(qr(X0).Q)
        end
    end

    for (ik, kpt) in enumerate(kpoints)
        results[ik] = kernel(kblock(ham, kpt), get_guessk(guess, ik);
                             prec=kblock(prec, kpt), tol=tol, maxiter=maxiter,
                             kwargs...)
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

"""
DOCME
"""
function construct_diag(kernel; kwargs...)
    # Return a function, which calls the diagonalisation_kloop routine.
    # By default the kwargs from the scf (passed as scfkwargs) are used,
    # unless they are overwritten by the kwargs passed upon call to diag_lobpcg.
    (ham, n_ep; scfkwargs...) -> diagonalise_all_kblocks(kernel, ham, n_ep;
                                                         merge(scfkwargs, kwargs)...)
end

# The actual implementations using the above primitives
include("diag_lobpcg_scipy.jl")
include("diag_lobpcg_hyper.jl")
include("diag_lobpcg_itsolve.jl")
