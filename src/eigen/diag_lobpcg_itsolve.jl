import IterativeSolvers

function lobpcg_itsolve(A, X0; prec=nothing, tol, maxiter, kwargs...)
    @warn "lobpcg_itsolve is not tested. Use at your own risk."
    res = IterativeSolvers.lobpcg(A, false, X0; P=prec, tol=tol, maxiter=maxiter)
    (λ=res.λ, X=res.X,
     residual_norms=res.residual_norms,
     iterations=res.iterations,
     converged=res.converged,
     n_matvec=0)
end
