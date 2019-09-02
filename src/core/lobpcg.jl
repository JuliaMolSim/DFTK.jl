include("lobpcg_itsolve.jl")
include("lobpcg_scipy.jl")
include("lobpcg_hyper.jl")

# These wrapper structures are needed to get things working properly
# with the LOBPCG backends we use.
struct HamiltonianBlock
    ham::Hamiltonian
    pot_hartree_values
    pot_xc_values
    ik::Int
end
Base.size(block::HamiltonianBlock, idx::Int) = length(block.ham.basis.basis_wf[block.ik])
Base.size(block::HamiltonianBlock) = (size(block, 1), size(block, 2))
Base.eltype(block::HamiltonianBlock) = eltype(block.ham)
function LinearAlgebra.mul!(out_Xk, block::HamiltonianBlock, in_Xk)
    return apply_hamiltonian!(out_Xk, block.ham, block.ik, block.pot_hartree_values,
                              block.pot_xc_values, in_Xk)
end
import Base: *, \
*(block::HamiltonianBlock, in_Xk) = mul!(similar(in_Xk), block, in_Xk)
struct PreconditionerBlock
    precond
    ik::Int
end
LinearAlgebra.ldiv!(Y, block::PreconditionerBlock, B) = ldiv!(Y, block.precond, block.ik, B)
\(block::PreconditionerBlock, B) = ldiv!(similar(B), block, B)

"""
Interpolate some data from one k-Point to another. The interpolation is fast, but not
necessarily exact or even normalised. Intended only to construct guesses for iterative
solvers
"""
function interpolate_at_kpoint(pw, old_ik, new_ik, data_oldk::AbstractVecOrMat)
    @assert 0 <= new_ik <= length(pw.kpoints)
    @assert 0 <= old_ik <= length(pw.kpoints)
    if new_ik == old_ik
        return data_oldk
    end

    basis_old = pw.basis_wf[old_ik]
    basis_new = pw.basis_wf[new_ik]
    @assert length(basis_old) == size(data_oldk, 1)
    n_bands= size(data_oldk, 2)

    data_newk = similar(data_oldk, length(basis_new), n_bands) .= 0
    for (iold, inew) in enumerate(indexin(basis_old, basis_new))
        inew !== nothing && (data_newk[inew, :] = data_oldk[iold, :])
    end
    data_newk
end

@doc raw"""
    lobpcg(ham::Hamiltonian, pot_hartree_values, pot_xc_values, nev_per_kpoint::Int;
           guess=nothing, prec=nothing, tol=1e-6, maxiter=200, backend=:lobpcg_hyper,
           kwargs...)

Run the LOBPCG implementation from `backend` for each ``k``-Point of the Hamiltonian `ham`,
solving for the `nev_per_kpoint` smallest eigenvalues. Optionally a `guess` and a
preconditioner `prec` can be used. `pot_hartree_values` and `pot_xc_values` are the
precomputed values of the Hartee and XC term of the Hamiltonian on the grid ``B_ρ^∗``.
The `backend` parameters selects the LOBPCG implementation to use. If no guess is supplied
and `interpolate_kpoints` is true, the function tries to interpolate the results of
earlier ``k``-Points to be used as guesses for the later ones.
"""
function lobpcg(ham::Hamiltonian, nev_per_kpoint::Int;
                pot_hartree_values=nothing, pot_xc_values=nothing,
                guess=nothing, prec=nothing, tol=1e-6, maxiter=200,
                backend=:lobpcg_hyper, interpolate_kpoints=true, kwargs...)
    if !(backend in [:lobpcg_itsolve, :lobpcg_scipy, :lobpcg_hyper])
        error("LOBPCG backend $(string(backend)) unknown.")
    end

    T = eltype(ham)
    pw = ham.basis
    n_kpoints = length(pw.kpoints)
    results = Vector{Any}(undef, n_kpoints)

    # Function simplifying stuff to be done for each k-Point
    get_hamk(ik) = HamiltonianBlock(ham, pot_hartree_values, pot_xc_values, ik)
    get_preck(::Nothing, ik) = nothing
    get_preck(prec, ik) = PreconditionerBlock(prec, ik)
    function get_guessk(guess, ik)
        # TODO Proper error messages
        @assert length(guess) ≥ n_kpoints
        @assert size(guess[ik], 2) == nev_per_kpoint
        @assert size(guess[ik], 1) == length(pw.basis_wf[ik])
        guess[ik]
    end
    function get_guessk(::Nothing, ik)
        if ik <= 1 || !interpolate_kpoints
            # TODO The double conversion is needed due to an issue in Julia
            #      see https://github.com/JuliaLang/julia/pull/32979
            qrres = qr(randn(real(T), length(pw.basis_wf[ik]), nev_per_kpoint))
            m = Matrix{T}(Matrix(qrres.Q))
        else  # Interpolate from previous k-Point
            X0 = interpolate_at_kpoint(pw, ik - 1, ik, results[ik - 1].X)
            X0 = Matrix{T}(qr(X0).Q)
        end
    end

    # Lookup "backend" symbol and use the corresponding function as run_lobpcg
    run_lobpcg = getfield(DFTK, backend)
    for ik in 1:n_kpoints
        results[ik] = run_lobpcg(get_hamk(ik), get_guessk(guess, ik);
                                 prec=get_preck(prec, ik), tol=tol, maxiter=maxiter,
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
