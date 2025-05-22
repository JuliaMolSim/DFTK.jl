using LinearMaps
using KrylovKit: OrthonormalBasis, Orthogonalizer
using LinearAlgebra: Givens

# Perform an inexact matrix-vector product, ensuring that
# maximum(norm, eachcol(Y - A*X)) < tol
inexact_mul(A, x; tol=0.0) = (; Ax=A * x, info=(; tol))

function default_gmres_print(info)
    !mpi_master() && return info  # Rest is printing => only do on master

    n_iter = info.n_iter
    if info.stage == :iterate
        if n_iter == 1
            @printf("%4s  %6s  %6s  %15s  %15s\n", "iter", "restrt", "krydim", "residual", "comment")
        end
        comment = ""
        if ((n_iter-1) in info.restart_history)
            comment = "Restart"
        end
        @printf("%4i  %6i  %6i  %15.8g  %15s\n", n_iter, length(info.restart_history),
                info.k, info.resid_history[n_iter], comment)
    end
    info
end

@doc raw"""
Preconditioned Inexact GMRES algorithm as discussed in [arxiv 2505.02319](https://arxiv.org/pdf/2505.02319).
At convergence ``\| \text{precon}^{-1} (A*x - b) \| < \text{tol}`` is ensured.

Overview of parameters:
- **krylovdim**: Maximal Krylov subspace dimension before restart
- **tol**:  Convergence threshold
- **s**: Initial guess for the smallest singular value of the upper Hessenberg matrix.
- **orth**: `KrylovKit.Orthogonalizer`: Orthogonalisation routine
- **callback**: Callback function
- **precon**: Left preconditioner
"""
function inexact_gmres!(x, A, b::AbstractVector{T};
                        precon=I, maxiter=100, krylovdim=20, tol=1e-6, s=1.0, callback=identity,
                        orth::Orthogonalizer=KrylovKit.ModifiedGramSchmidt2()) where {T}
    m = krylovdim
    r = zero(b)  # Storage for residual vector
    w = zero(b)  # Storage for intermediates (precon\Ax or precon\r)

    # Arnoldi subspace variables:
    H  = zeros(T, m + 1, m)     # Hessenberg matrix  (a bit wasteful ... stores all zeros)
    y  = zeros(T, m + 1)        # Solution in Arnoldi basis
    V  = OrthonormalBasis{typeof(b)}()  # Arnoldi basis
    sizehint!(V, m)

    # For solving the least-squares problem, we find a QR decomposition of H,
    # i.e. an orthogonal matrix Q and R, s.t. Q H = R.
    # We represent Q as a product of Givens rotations
    Gs = Vector{Givens{T}}(undef, m)  # Vector of Givens rotations making the
    R  = zeros(T, m, m)               # upper-triangular matrix

    resid_history = zeros(real(T), maxiter)  # Residual norms
    restart_history = Int[]                  # Indices where restart has occurred
    Axinfo = (; )

    converged = false
    n_iter = 0
    while true  # Loop for restarts
        @assert isempty(V)
        k  = 0   # Number of Krylov vectors
        if iszero(x) && n_iter == 0
            ldiv!(r, precon, b)  # Apply preconditioner
        else
            Ax, Axinfo = inexact_mul(A, x; tol=tol/3)
            w .= b .- Ax
            ldiv!(r, precon, w)  # Apply preconditioner
        end
        y[1] = β = norm(r)

        info = (; y, V, H, R, resid_history=resid_history[1:n_iter], converged, n_iter,
                 residual_norm=β, maxiter, tol, s, residual=r,
                 restart_history, stage=:restart, krylovdim, k, Axinfo)

        while (n_iter < maxiter && k < m)  # Arnoldi loop
            n_iter += 1

            # Push new Krylov vector
            push!(V, r / β)
            k = length(V)

            # Compute new Krylov vector and orthogonalise against subspace
            tolA = tol * s / (3m * abs(y[k]))  # |y[k]| is the estimated residual norm
            p, Axinfo = inexact_mul(A, V[k]; tol=tolA)
            ldiv!(w, precon, p)  # Apply preconditioner
            r, _ = orthogonalize!!(w, V, @view(H[1:k, k]), orth)
            H[k+1, k] = β = norm(r)

            # Copy Hessenberg matrix into R[:, k] and apply the collected Givens rotations
            R[1:k, k] = H[1:k, k]
            @inbounds for i in 1:(k-1)
                lmul!(Gs[i], @view(R[:, k]))
            end
            Gs[k], R[k, k] = givens(R[k, k], H[k+1, k], k, k+1)

            # Update right-hand side in Krylov subspace and new residual norm estimate
            y[k + 1] = zero(T)
            lmul!(Gs[k], y)
            resid_history[n_iter] = abs(y[k + 1])

            info = (; y, V, H, R, resid_history=resid_history[1:n_iter], converged, n_iter,
                     residual_norm=resid_history[n_iter], maxiter, tol, s, residual=r,
                     restart_history, stage=:iterate, krylovdim, k, Axinfo)
            callback(info)

            if resid_history[n_iter] < tol
                # If the guess for s happens to over-estimate the σ(H_m) than we need to
                # restart, so convergence is only reached if this condition is true ...
                min_svdval_H = minimum(svdvals(H[1:(k+1), 1:k]))
                converged = s < min_svdval_H
                break  # ... but we restart in any case
            end
        end

        # Update x by solving upper triangular system
        @views ldiv!(UpperTriangular(R[1:k, 1:k]), y[1:k])
        @inbounds for i in 1:k
            x += y[i] * V[i]
        end

        if converged || n_iter ≥ maxiter
            info = (; x, resid_history=resid_history[1:n_iter], converged, n_iter,
                     residual_norm=resid_history[n_iter], maxiter, tol, s, residual=r,
                     restart_history, stage=:finalize, krylovdim, y, V, H, R)
            callback(info)
            return info
        end

        min_svdval_H = minimum(svdvals(H[1:(k+1), 1:k]))
        @debug("  Restart: " * (k ≥ krylovdim    ? "Max Krylov space reached" :
                               (s ≥ min_svdval_H ? "s too large"              : "")))

        # Update guess for lowest singular value of H and restart
        s = min(s, min_svdval_H)
        empty!(V)
        push!(restart_history, n_iter)
    end
end
function inexact_gmres(A, b; kwargs...)
    inexact_gmres!(zeros_like(b, eltype(b), size(A, 2)), A, b; kwargs...)
end
