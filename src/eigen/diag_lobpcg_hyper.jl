include("lobpcg_hyper_impl.jl")

function lobpcg_hyper(A, X0; maxiter=100, prec=nothing,
                      tol=20size(A, 2)*eps(real(eltype(A))),
                      largest=false, n_conv_check=nothing, kwargs...)
    prec === nothing && (prec = I)

    @assert !largest "Only seeking the smallest eigenpairs is implemented."
    result = LOBPCG(A, X0, I, prec, tol, maxiter; n_conv_check=n_conv_check, kwargs...)

    n_conv_check === nothing && (n_conv_check = size(X0, 2))

    converged = maximum(result.residual_norms[1:n_conv_check]) < tol
    iterations = size(result.residual_history, 2) - 1
    λ = Array(result.λ) #TODO: offload this to gpu? Careful then, as self_consistent_field's eigenvalues will be a CuArray -> due to the Smearing.occupation function, occupation will also be a CuArray, so no scalar indexing (in ene_ops, in compute_density...)

    merge(result, (iterations=iterations, converged=converged, λ=λ))
end
