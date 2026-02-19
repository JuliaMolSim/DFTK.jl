using LinearMaps
using KrylovKit: OrthonormalBasis, Orthogonalizer
using LinearAlgebra: Givens

"""
Matrix-vector product function used by `inexact_gmres`. The intention is that an
approximate matrix-vector product `Y = A*X` is computed, however, ensuring that each
matvec is accurate up to a relative tolerance `rtol`,
i.e. `norm(Y[:, i] - A*X[:, i]) < rtol * norm(X[:, i])`. Specific matrix types `A` may
define a method for this function to provide faster approximate versions for `A*X`.
"""
mul_approximate(A, X; rtol=0.0) = (; Ax=A * X, info=(; rtol))

function default_gmres_print(info)
    # Default callback prints on all MPI ranks
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
Solve a linear system `Ax=b` involving operator `A` by a preconditioned inexact GMRES algorithm.
At convergence ``\| \text{precon}^{-1} (A*x - b) \| < \text{tol} * \|b\|`` is ensured.

Like standard GMRES the algorithm builds a Krylov subspace and uses the Arnoldi relationship
`A V = V H` where `V` are the Arnoldi vectors and `H` is an upper Hessenberg matrix. Based
on this decomposition the projection of the linear problem within the Krylov subspace can be
efficiently solved. In contrast to standard GMRES methods, however, the inexact GMRES tolerates
the Arnoldi relationship to hold only approximately, which means that in turn matrix-vector
products `A*v` do not all need to be fully accurate. More precisely, since the coefficients
of the solution vector `x`, when expanded in the basis `V` *decreases* from one Arnoldi
vector to the next, `A*v` can be computed *less and less precise* as the GMRES is converging,
see [^Simoncini2003] for details. The implementation employs the function `mul_approximate`
to signal the operator how accurate a particular matvec is needed. For more details on our
implementation see Algorithm 3.1. of [^Herbst2025].

[^Simoncini2003]: [DOI 10.1137/s1064827502406415](http://dx.doi.org/10.1137/s1064827502406415)
[^Herbst2025]: [arxiv 2505.02319](https://arxiv.org/pdf/2505.02319)

Standard GMRES parameters:
- **x**:         Initial guess vector
- **krylovdim**: Maximal Krylov subspace dimension before restart
- **tol**:       Absolute convergence threshold
- **orth**:      `KrylovKit.Orthogonalizer`: Orthogonalisation routine
- **callback**:  Callback function
- **precon**:    Left preconditioner

Parameters specific to inexact GMRES:
- **s**: Initial guess for the smallest singular value of the upper Hessenberg matrix;
         will be adapted on the fly.
"""
@timing function inexact_gmres!(x, A, b::AbstractVector{T};
                                precon=I, maxiter=100, krylovdim=20, tol=1e-6, s=1.0,
                                callback=identity,
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
    Axinfos = []

    converged = false
    n_iter = 0
    while true  # Loop for restarts
        @assert isempty(V)
        k  = 0   # Number of Krylov vectors
        if iszero(x) && n_iter == 0
            ldiv!(r, precon, b)  # Apply preconditioner
        else
            Ax, Axinfo = mul_approximate(A, x; rtol=tol/3/norm(x))
            push!(Axinfos, Axinfo)
            w .= b .- Ax
            ldiv!(r, precon, w)  # Apply preconditioner
        end
        y[1] = β = residual_norm = norm(r)

        # Prevent iterations when the initial guess or the restarted guess are already sufficiently
        # accurate. We check residual_norm < 2/3 tol since ||b-Ax|| < ||b-Ãx|| + ||Ax-Ãx||,
        # where Ã is the inexact operator used in mul_approximate, which is accurate to tol/3.
        converged = residual_norm < 2tol/3

        while (!converged && n_iter < maxiter && k < m)  # Arnoldi loop
            n_iter += 1

            # Push new Krylov vector
            push!(V, r / β)
            k = length(V)

            # Compute new Krylov vector and orthogonalise against subspace
            tolA = tol * s / (3m * abs(y[k]))  # |y[k]| is the estimated residual norm
            p, Axinfo = mul_approximate(A, V[k]; rtol=tolA)  # ||V[k]|| = 1
            push!(Axinfos, Axinfo)
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
            resid_history[n_iter] = residual_norm = abs(y[k + 1])

            info = (; y, V, H, R, resid_history=resid_history[1:n_iter], converged, n_iter,
                     residual_norm, maxiter, tol, s,
                     restart_history, stage=:iterate, krylovdim, k, Axinfos)
            callback(info)
            Axinfos = []

            # Note: Although in [^Herbst2025] we have tol / 3 in the following convergence check,
            # our numerical tests still find that the requested tolerance is approximately
            # achieved when dropping the /3. This prevents the GMRES from potential over-iterating
            # when the `mul_approximate` calls happen to yield higher-accurate answers compared to
            # what has been requested.
            if residual_norm < tol
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
                     residual_norm, maxiter, tol, s,
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
