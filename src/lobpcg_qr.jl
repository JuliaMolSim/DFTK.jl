using LinearAlgebra

"""
A very simple LOBPCG for finding the largest eigenpairs of non-general eigenproblems
"""
function lobpcg_qr(A, X0; maxiter=100, Prec=I, tol=20size(A,2)*eps(eltype(A)))
    ortho(X) = Array(qr(X).Q)

    N = size(A, 2)
    m = size(X0, 2)
    T = eltype(A)

    # Allocate containers for subspace data
    Y = similar(X0, T, N, 3m)
    X = view(Y, :, 1:m)
    R = similar(X0, T, N, m)
    P = view(Y, :, 2m+1:3m) .= 0

    # Storage for A*Y
    Ay = similar(Y, T)

    # Orthogonalise X0 and apply A to it.
    X .= ortho(X0)
    rvals = zeros(T, m, m)
    converged = false
    mul!(view(Ay, :, 1:m), A, X)
    niter = 1
    while niter <= maxiter
        # Compute residuals and Ritz values
        rvals .= X' * Ay[:, 1:m]
        R .= Ay[:, 1:m] - X * rvals
        if norm(R) < tol
            converged = true
            break
        end
        niter += 1

        # Update the residual slot of Y
        if Prec == I || Prec === nothing
            Y[:, m+1:2m] .= R
        else
            ldiv!(view(Y, :, m+1:2m), Prec, R)
        end

        # Orthogonalise Y and solve Rayleigh-Ritz step
        Y .= ortho(Y)
        mul!(Ay, A, Y)

        c = eigvecs(Hermitian(Y'*Ay))[:, 1:m]
        new_X = Y*c
        P .= new_X - X
        X .= new_X
        Ay[:, 1:m] .= Ay * c
    end

    (λ=real(diag(rvals)),
     X=X,
     residual_norms=[norm(R[:, i]) for i in 1:m],
     iterations=niter,
     converged=converged)
end
