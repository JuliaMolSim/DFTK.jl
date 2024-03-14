include("lobpcg_hyper_impl.jl")

function lobpcg_hyper(A, X0; maxiter=100, prec=nothing,
                      tol=20size(A, 2)*eps(real(eltype(A))),
                      largest=false, n_conv_check=nothing, kwargs...)
    prec === nothing && (prec = I)

    @assert !largest "Only seeking the smallest eigenpairs is implemented."
    result = LOBPCG(A, X0, I, prec, tol, maxiter; n_conv_check, kwargs...)

    n_conv_check === nothing && (n_conv_check = size(X0, 2))
    converged = maximum(result.residual_norms[1:n_conv_check]) < tol
    n_iter = size(result.residual_history, 2) - 1

    merge(result, (; n_iter, converged))
end
