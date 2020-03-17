import IterativeSolvers

function lobpcg_itsolve(A, X0; prec=nothing, tol, maxiter, kwargs...)
    @warn "lobpcg_itsolve is not tested. Use at your own risk."
    IterativeSolvers.lobpcg(A, false, X0; P=prec, tol=tol, maxiter=maxiter)
end
