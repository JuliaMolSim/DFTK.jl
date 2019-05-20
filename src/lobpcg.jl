using IterativeSolvers
import IterativeSolvers: LOBPCGResults
import IterativeSolvers: LOBPCGState
import IterativeSolvers: lobpcg


# Setup HamiltonianBlock struct and define the required functions
# for it to be working with the lobpcg function from IterativeSolvers
struct HamiltonianBlock
    ham::Hamiltonian
    precomp_hartree
    precomp_xc
    ik::Int
end

Base.size(block::HamiltonianBlock, idx::Int) = length(block.ham.basis.kmask[block.ik])
Base.eltype(block::HamiltonianBlock) = eltype(block.ham)
function LinearAlgebra.mul!(out_Xk, block::HamiltonianBlock, in_Xk)
    return apply_fourier!(out_Xk, block.ham, block.ik, block.precomp_hartree,
                          block.precomp_xc, in_Xk)
end

# Setup PreconditionerBlock struct and define the required functions
struct PreconditionerBlock
    precond
    ik::Int
end
function LinearAlgebra.ldiv!(Y, block::PreconditionerBlock, B)
    apply_inverse_fourier!(Y, block.precond, block.ik, B)
end

"""
TODO Docme
"""
function lobpcg(ham::Hamiltonian, nev_per_kpoint::Int;
                precomp_hartree=nothing, precomp_xc=nothing,
                largest=false, guess=nothing, preconditioner=nothing, tol=1e-6,
                maxiter=200, kwargs...)
    # TODO Trigger precomp computation if passed object is nothing

    T = eltype(ham)
    pw::PlaneWaveBasis = ham.basis
    n_k = length(pw.kpoints)

    # TODO This interface is in general really ugly and should be reworked
    res = Dict{Symbol, Any}(
        :λ => Vector{Vector{T}}(undef, n_k),
        :X => Vector{Matrix{T}}(undef, n_k),
        :residual_norms => Vector{Vector{T}}(undef, n_k),
        :iterations => Vector{Int}(undef, n_k),
        :converged => true,
    )

    for ik in 1:n_k
        Pk = nothing
        if preconditioner !== nothing
            Pk = PreconditionerBlock(preconditioner, ik)
        end

        hamk = HamiltonianBlock(ham, precomp_hartree, precomp_xc, ik)

        itres = nothing
        if guess === nothing
            itres = lobpcg(hamk, largest, nev_per_kpoint, P=Pk,
                           tol=tol, maxiter=maxiter, kwargs...)
        else
            # TODO Proper error messages
            @assert length(guess) ≥ n_k
            @assert size(guess[ik], 2) == nev_per_kpoint
            @assert size(guess[ik], 1) == size(hamk, 2)
            itres = lobpcg(hamk, largest, guess[ik],
                           P=Pk, tol=tol, maxiter=maxiter, kwargs...)
        end

        # Add iteration result to res:
        res[:λ][ik]              = itres.λ
        res[:X][ik]              = itres.X
        res[:residual_norms][ik] = itres.residual_norms
        res[:iterations][ik]     = itres.iterations
        res[:converged]          = itres.converged && res[:converged]
    end

    LOBPCGResults(res[:λ], res[:X], tol, res[:residual_norms], res[:iterations],
                  maxiter, res[:converged],
                  Vector{LOBPCGState{Vector{Vector{T}}, Vector{Vector{T}}}}(undef, 0))
end
