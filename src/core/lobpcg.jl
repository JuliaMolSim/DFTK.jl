include("lobpcg_itsolve.jl")
include("lobpcg_scipy.jl")
include("lobpcg_qr.jl")

# These wrapper structures are needed to get things working properly
# with the LOBPCG backends we use.
struct HamiltonianBlock
    ham::Hamiltonian
    pot_hartree_values
    pot_xc_values
    ik::Int
end
Base.size(block::HamiltonianBlock, idx::Int) = length(block.ham.basis.basis_wf[block.ik])
Base.eltype(block::HamiltonianBlock) = eltype(block.ham)
function LinearAlgebra.mul!(out_Xk, block::HamiltonianBlock, in_Xk)
    return apply_hamiltonian!(out_Xk, block.ham, block.ik, block.pot_hartree_values,
                              block.pot_xc_values, in_Xk)
end
struct PreconditionerBlock
    precond
    ik::Int
end
LinearAlgebra.ldiv!(Y, block::PreconditionerBlock, B) = ldiv!(Y, block.precond, block.ik, B)


@doc raw"""
    lobpcg(ham::Hamiltonian, pot_hartree_values, pot_xc_values, nev_per_kpoint::Int;
           guess=nothing, prec=nothing, tol=1e-6, maxiter=100, backend=:lobpcg_qr,
           kwargs...)

Run the LOBPCG implementation from `backend` for each ``k``-Point of the Hamiltonian `ham`,
solving for the `nev_per_kpoint` smallest eigenvalues. Optionally a `guess` and a
preconditioner `prec` can be used. `pot_hartree_values` and `pot_xc_values` are the
precomputed values of the Hartee and XC term of the Hamiltonian on the grid ``B_ρ^∗``.
The `backend` parameters selects the LOBPCG implementation to use.
"""
function lobpcg(ham::Hamiltonian, nev_per_kpoint::Int;
                pot_hartree_values=nothing, pot_xc_values=nothing,
                guess=nothing, prec=nothing, tol=1e-6, maxiter=200,
                backend=:auto, kwargs...)
    if backend == :auto
        # First try IterativeSolvers.jl, then our QR fallback implementation
        try
            return lobpcg(ham, nev_per_kpoint; pot_hartree_values=pot_hartree_values,
                          pot_xc_values=pot_xc_values, guess=guess, prec=prec,
                          tol=tol, maxiter=maxiter, backend=:lobpcg_itsolve, kwargs...)
        catch PosDefException
            return lobpcg(ham, nev_per_kpoint; pot_hartree_values=pot_hartree_values,
                          pot_xc_values=pot_xc_values, guess=guess, prec=prec,
                          tol=tol, maxiter=maxiter, backend=:lobpcg_qr, kwargs...)
        end
    end

    if !(backend in [:lobpcg_itsolve, :lobpcg_qr, :lobpcg_scipy])
        error("LOBPCG backend $(str(backend)) unknown.")
    end

    # TODO This function seems to be type-unstable ... check
    #
    T = eltype(ham)
    pw = ham.basis

    # Function simplifying stuff to be done for each k-Point
    get_hamk(ik) = HamiltonianBlock(ham, pot_hartree_values, pot_xc_values, ik)
    get_guessk(::Nothing, ik) = Matrix(qr(randn(real(T), length(pw.basis_wf[ik]),
                                                nev_per_kpoint)).Q)
    function get_guessk(guess, ik)
        # TODO Proper error messages
        @assert length(guess) ≥ length(pw.kpoints)
        @assert size(guess[ik], 2) == nev_per_kpoint
        @assert size(guess[ik], 1) == length(pw.basis_wf[ik])
        guess[ik]
    end
    get_preck(::Nothing, ik) = nothing
    get_preck(prec, ik) = PreconditionerBlock(prec, ik)

    # Lookup "backend" symbol and use the corresponding function as run_lobpcg
    run_lobpcg = getfield(DFTK, backend)
    results = [run_lobpcg(get_hamk(ik), get_guessk(guess, ik), prec=get_preck(prec, ik),
                          tol=tol, maxiter=maxiter, kwargs...)
               for ik in 1:length(pw.kpoints)]

    # Transform results into a nicer datastructure
    (λ=[real.(res.λ) for res in results],
     X=[res.X for res in results],
     residual_norms=[res.residual_norms for res in results],
     iterations=[res.iterations for res in results],
     converged=all(res.converged for res in results),
     implementation=string(backend))
end
