const lobpcg_backend = :lobpcg_qr

if lobpcg_backend == :scipy
    using PyCall
elseif lobpcg_backend == :IterativeSolvers
    using IterativeSolvers
    import IterativeSolvers: lobpcg, PosDefException
elseif lobpcg_backend == :lobpcg_qr
    include("lobpcg_qr.jl")
end

if lobpcg_backend == :scipy
    """
    Call scipy's `lobpcg` function on an operator `hamk` using the guess `X0`,
    which also defines the block size. `P` is an optional preconditioner.
    If `largest` is true, the largest eigenvalues will be sought, else the
    smallest. Both `hamk` as well as `P` will be transformed to
    `scipy.sparse.LinearOperator` along the call.
    """
    function lobpcg(hamk, largest::Bool, X0; P=nothing, tol=nothing, kwargs...)
        sla = pyimport_conda("scipy.sparse.linalg", "scipy")

        @assert size(X0, 1) == size(hamk, 2)
        @assert eltype(hamk) == ComplexF64
        A = sla.LinearOperator((size(hamk, 1), size(hamk, 2)),
                               matvec=(v -> mul!(similar(v, ComplexF64), hamk, v)),
                               dtype="complex128")
        M = nothing
        if P !== nothing
            M = sla.LinearOperator((size(hamk, 1), size(hamk, 2)),
                                   matvec=(v -> ldiv!(similar(v, ComplexF64), P, v)),
                                   dtype="complex128")
        end

        if tol !== nothing
            tol /= 10
        end

        res = sla.lobpcg(A, X0, M=M, retResidualNormsHistory=true; tol=tol,
                         largest=largest, kwargs...)
        λ = real(res[1])
        order = sortperm(λ)  # Order to sort eigenvalues ascendingly
        maxnorm = maximum(real(res[3][end][order]))

        is_converged = true
        if maxnorm !== nothing
            is_converged = maxnorm < 10 * tol
        end

        (λ=λ[order],
         X=res[2][:, order],
         residual_norms=real(res[3][end][order]),
         iterations=length(res[3]),
         converged=is_converged)
    end


    """
    Call scipy's `lobpcg` function on an operator `hamk`, solving for
    `nev` eigenvalues. If `largest` is true, the largest eigenvalues will
    be sought, else the smallest. Both `hamk` as well as `P` will be
    transformed to `scipy.sparse.LinearOperator` along the call.
    """
    function lobpcg(hamk, largest::Bool, nev::Int; kwargs...)
        X0 = randn(real(eltype(hamk)), size(hamk, 2), nev)
        lobpcg(hamk, largest, Matrix(qr(X0).Q); kwargs...)
    end
elseif lobpcg_backend == :lobpcg_qr
    function lobpcg(hamk, largest::Bool, X0; P=nothing, kwargs...)
        @assert !largest  # only smallest implemented
        lobpcg_qr(hamk, X0; Prec=P, kwargs...)
    end

    function lobpcg(hamk, largest::Bool, nev::Int; kwargs...)
        X0 = randn(real(eltype(hamk)), size(hamk, 2), nev)
        lobpcg(hamk, largest, Matrix(qr(X0).Q); kwargs...)
    end
end


# Setup HamiltonianBlock struct and define the required functions
# for it to be working with the lobpcg function from IterativeSolvers
struct HamiltonianBlock
    ham::Hamiltonian
    precomp_hartree
    precomp_xc
    ik::Int
end

Base.size(block::HamiltonianBlock, idx::Int) = length(block.ham.basis.wf_basis[block.ik])
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

    # TODO λ and X are not the best names
    converged = true
    res = (λ=Vector{Vector{real(T)}}(undef, n_k),
           X=Vector{Matrix{T}}(undef, n_k),
           residual_norms=Vector{Vector{real(T)}}(undef, n_k),
           iterations=Vector{Int}(undef, n_k),
           implementation=string(lobpcg_backend),
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
        res.λ[ik]              = itres.λ
        res.X[ik]              = itres.X
        res.residual_norms[ik] = itres.residual_norms
        res.iterations[ik]     = itres.iterations
        converged              = converged && itres.converged
    end
    return merge(res, (converged=converged, ))
end
