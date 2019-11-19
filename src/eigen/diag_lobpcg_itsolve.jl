import IterativeSolvers

lobpcg_itsolve(A, X0; prec=nothing, tol, maxiter) = IterativeSolvers.lobpcg(A, false, X0; P=prec, tol=tol, maxiter=maxiter)
