include("lobpcg_itsolve.jl")
include("lobpcg_scipy.jl")
include("lobpcg_hyper.jl")

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
    lobpcg(ham::Hamiltonian, nev_per_kpoint::Int;
           guess=nothing, prec=nothing, tol=1e-6, maxiter=200, backend=:lobpcg_hyper,
           kwargs...)

Run the LOBPCG implementation from `backend` for each ``k``-Point of the Hamiltonian `ham`,
solving for the `nev_per_kpoint` smallest eigenvalues. Optionally a `guess` and a
preconditioner `prec` can be used. The `backend` parameters selects the LOBPCG
implementation to use. If no guess is supplied and `interpolate_kpoints` is true,
the function tries to interpolate the results of earlier ``k``-Points to be used as guesses
for the later ones.
"""
# TODO Also use function-like interface here (e.g. eigensolver=lobpcg_hyper(stuff...)
function lobpcg(ham::Hamiltonian, nev_per_kpoint::Int, kpoints=ham.basis.kpoints;
                guess=nothing, prec=nothing, tol=1e-6, maxiter=200,
                backend=:lobpcg_hyper, interpolate_kpoints=true, kwargs...)
    if !(backend in [:lobpcg_itsolve, :lobpcg_scipy, :lobpcg_hyper])
        error("LOBPCG backend $(string(backend)) unknown.")
    end
    # TODO The interface of this function should be thought through now.
    #      Also keeping in mind that we now have a specialised function for band structure
    #      calculations.

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

    # Lookup "backend" symbol and use the corresponding function as run_lobpcg
    run_lobpcg = getfield(DFTK, backend)
    for (ik, kpt) in enumerate(kpoints)
        results[ik] = run_lobpcg(kblock(ham, kpt), get_guessk(guess, ik);
                                 prec=kblock(prec, kpt), tol=tol, maxiter=maxiter,
                                 kwargs...)
    end

    # Transform results into a nicer datastructure
    (λ=[real.(res.λ) for res in results],
     X=[res.X for res in results],
     residual_norms=[res.residual_norms for res in results],
     iterations=[res.iterations for res in results],
     converged=all(res.converged for res in results),
     backend=string(backend))
end
