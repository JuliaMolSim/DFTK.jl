using PyCall

"""
    lobpcg_scipy(A, X0; prec=nothing, tol=nothing, largest=false, kwargs...)

Call scipy's version of LOBPCG for finding the eigenpairs of a Hermitian matrix `A`.
`X0` is the set of guess vectors, also determining the number of eigenpairs to be sought.
"""
function lobpcg_scipy(A, X0; prec=nothing, tol, maxiter, largest=false)
    sla = pyimport("scipy.sparse.linalg")
    sys = pyimport("sys")

    @assert size(X0, 1) == size(A, 2)
    @assert eltype(A) == ComplexF64
    opA = sla.LinearOperator((size(A, 1), size(A, 2)),
                           matvec=(v -> mul!(similar(v, ComplexF64), A, v)),
                           dtype="complex128")
    opP = nothing
    if prec !== nothing
        opP = sla.LinearOperator((size(A, 1), size(A, 2)),
                                 matvec=(v -> ldiv!(similar(v, ComplexF64), prec, v)),
                                 dtype="complex128")
    end

    res = sla.lobpcg(opA, X0, M=opP, retResidualNormsHistory=true; tol=tol,
                     largest=largest, maxiter=maxiter)
    sys.stdout.flush()
    λ = real(res[1])  # 1 is eigenvalues
    order = sortperm(λ)  # Order to sort eigenvalues ascendingly
    maxnorm = maximum(real(res[3][end][order]))  # 3 is residual norms

    converged = true
    if maxnorm !== nothing
        converged = maxnorm < 30 * tol
    end

    (λ=λ[order],
     X=res[2][:, order],
     residual_norms=real(res[3][end][order]),
     iterations=length(res[3]),
     converged=converged)
end
