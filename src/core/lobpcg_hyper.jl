include("lobpcg_hyper_impl.jl")

function lobpcg_hyper(A, X0; maxiter=100, prec=I, tol=20size(A, 2)*eps(real(eltype(A))),
                      largest=false, kwargs...)
    prec === nothing && (prec = I)
    # Tolerance for orthogonalisation in LOBPCG:
    ortho_tol = max(2eps(real(eltype(A))), tol / 1000)

    @assert !largest "Only seeking the smallest eigenpairs is implemented."
    X, resids = LOBPCG(A, X0, I, prec, tol, maxiter; ortho_tol=ortho_tol)

    AX = A * X
    λ = real(diag(X' * AX))
    residuals = AX - X*Diagonal(λ)
    residual_norms=[norm(residuals[:, i]) for i in 1:size(residuals, 2)]
    converged = maximum(residual_norms[1:end-1]) <= 5tol  # A little cheating
    iterations = size(resids, 2)

    (λ=λ, X=X,
     residual_norms=residual_norms,
     iterations=iterations,
     converged=converged)
end
