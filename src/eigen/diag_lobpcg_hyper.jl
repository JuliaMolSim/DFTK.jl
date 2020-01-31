include("lobpcg_hyper_impl.jl")

function lobpcg_hyper(A, X0; maxiter=100, prec=nothing, tol=20size(A, 2)*eps(real(eltype(A))),
                      largest=false, n_conv_check=nothing, kwargs...)
    prec === nothing && (prec = I)

    @assert !largest "Only seeking the smallest eigenpairs is implemented."
    λ, X, residual_norms, resids = LOBPCG(A, X0, I, prec, tol, maxiter;
                                          n_conv_check=n_conv_check, kwargs...)

    n_conv_check === nothing && (n_conv_check = size(X0, 2))
    converged = maximum(residual_norms[1:n_conv_check]) < tol
    iterations = size(resids, 2)

    (λ=λ, X=X,
     residual_norms=residual_norms,
     iterations=iterations,
     converged=converged)
end
