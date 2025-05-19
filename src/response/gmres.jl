using LinearMaps
using KrylovKit: OrthonormalBasis, Orthogonalizer
using LinearAlgebra: Givens

# Perform an inexact matrix-vector product, ensuring that
# maximum(norm, eachcol(Y - A*X)) < tol
inexact_mul(A, x; tol=0.0) = A * x

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

    converged = false
    n_iter = 0
    while true  # Loop for restarts
        @assert isempty(V)
        k  = 0   # Number of Krylov vectors
        if iszero(x) && n_iter == 0
            ldiv!(r, precon, b)  # Apply preconditioner
        else
            Ax = inexact_mul(A, x; tol=tol/3)
            w .= b .- Ax
            ldiv!(r, precon, w)  # Apply preconditioner
        end
        y[1] = β = norm(r)

        while (n_iter < maxiter && k < m)  # Arnoldi loop
            n_iter += 1

            # Push new Krylov vector
            push!(V, r / β)
            k = length(V)

            # Compute new Krylov vector and orthogonalise against subspace
            tolA = tol * s / (3m * abs(y[k]))  # |y[k]| is the estimated residual norm
            p = inexact_mul(A, V[k]; tol=tolA)
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
                     restart_history, stage=:iterate, krylovdim, k)
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
    inexact_gmres!(zeros_like(b, eltype(b), size(A, 1)), A, b; kwargs...)
end










function gmres(operators::Function, b::AbstractVector{T}; 
               x₀=nothing,
               maxiter=min(100, size(operators(0), 1)),
               restart=min(20, size(operators(0), 1)),
               tol=1e-6, s_guess=0.8, verbose=1, debug=false) where {T}
    # pass s/3m 1/\|r_tilde\| tol to operators()
    normb = norm(b)
    b = b ./ normb
    tol = tol / normb
    if tol < 0.5*eps(T)
        #throw a warning
        @warn "The effective tolerance is smaller than half of the machine epsilon, the requested accuracy may not be achieved."
    end

    n_size = length(b)

    # counters
    numrestarts::Int64 = 0
    restart_inds = Int64[]
    MvTime = Float64[]

    # initialization: Arnoldi basis and Hessenberg matrix
    V = zeros(n_size, restart + 1)
    H = zeros(restart + 1, restart)

    # residual history
    residuals = zeros(maxiter)
    denominatorvec = zeros(restart)
    denominator2 = 1.0
    # record min singular values
    minsvdvals = zeros(maxiter)

    # solution history
    if debug
        X = zeros(n_size, maxiter)
    end

    # if x0 is zero, set r0 = b
    x0iszero = isnothing(x₀)
    if x0iszero
        β₀ = 1.0
        V[:, 1] = b #/ β₀
        tol_new = tol / 2
        lm = s_guess / restart / 2 * tol
    else
        # assert x₀ is a vector of the right size
        @assert length(x₀) == n_size
        x₀ ./= normb
        x0iszero = false
        tol_new = tol / 3
        A0_tol = tol / 3
        lm = s_guess / restart / 3 * tol
        push!(MvTime, @elapsed r₀ = operators(A0_tol) * x₀)
        r₀ = b - r₀
        β₀ = norm(r₀)
        V[:, 1] = r₀ / β₀
    end

    Ai_tol = lm / β₀ 
    i = 0 # iteration counter, set to 0 when restarted
    for j = 1:maxiter
        i += 1

        # Arnoldi iteration using MGS
        push!(MvTime, @elapsed w = operators(Ai_tol) * V[:, i])
        for j = 1:i
            H[j, i] = dot(V[:, j], w)
            w .-= H[j, i] * V[:, j]
        end
        H[i+1, i] = norm(w)
        V[:, i+1] = w / H[i+1, i]

        # compute the min singular value of the upper Hessenberg matrix
        minsvdvals[j] = svdvals(H[1:i+1, 1:i])[end]
    
        # update residual history
        if i == 1
            denominatorvec[1] = conj(-H[1, 1] / H[2, 1])
        else
            denominatorvec[i] = conj(-(H[1, i] + dot(denominatorvec[1:i-1], H[2:i, i])) / H[i+1, i])
        end
        denominator2 += abs2(denominatorvec[i])
        residuals[j] = β₀ / sqrt(denominator2)
        Ai_tol = lm / residuals[j]

        (verbose > 0) && println("restart = ", numrestarts + 1, ", iteration = ", i, ", res = ", residuals[j]*normb, "\n")

        # when converged without updating the s_guess, restart once
        converged = residuals[j] < tol_new
        happy = converged && ((numrestarts > 0) || (s_guess <= minsvdvals[j]))
        needrestart = (i == restart) || (converged && !happy) # equal to converged && (numrestarts == 0) && (s_guess > minsvdvals[j])
        reachmaxiter = (j == maxiter)
        exitloop = (happy || reachmaxiter)

        solvels = (exitloop || needrestart || debug)
        if solvels
            # solve the Hessenberg least squares problem
            rhs = (UpperTriangular(H[2:i+1, 1:i]) \ denominatorvec[1:i]) * (-β₀ / denominator2)
            if x0iszero
                xᵢ = V[:, 1:i] * rhs
            else
                xᵢ = x₀ + V[:, 1:i] * rhs
            end

            debug && (X[:, j] = xᵢ)
            
            if exitloop
                if debug
                    return (; x=X[:, 1:j].*normb, residuals=residuals[1:j].*normb, MvTime=MvTime, restart_inds=restart_inds, minsvdvals=minsvdvals[1:j])
                else
                    return (; x=xᵢ.*normb, residuals=residuals[1:j].*normb, MvTime=MvTime, restart_inds=restart_inds)
                end
            end

            if needrestart
                i = 0
                x0iszero = false
                # whenever restart, update the guess of the smallest singular value
                min_s = minimum(minsvdvals[1:j])
                if s_guess > min_s
                    verbose > 0 && println("current s_guess = ", s_guess, "> minsvdval = ", min_s, ", restart with s_guess = minsvdval")
                else
                    verbose > 0 && println("current s_guess = ", s_guess, "≤ minsvdval = ", min_s, ", restart with s_guess = minsvdval")
                end
                s_guess = min_s
                lm = s_guess / restart / 3 * tol
                tol_new = tol / 3
                A0_tol = tol / 3
                
                x₀ = xᵢ
                push!(MvTime, @elapsed r₀ = operators(A0_tol) * x₀)
                r₀ = b - r₀
                β₀ = norm(r₀)
                V[:, 1] = r₀ / β₀
                numrestarts += 1
                denominatorvec = zeros(maxiter)
                denominator2 = 1.0
                push!(restart_inds, j)
                (verbose > -1) && println("restarting: ", "iteration = ", j, ", r₀ = ", β₀, "\n")
                Ai_tol = lm / β₀ 
            end
        end
    end
end


function gmres(operator::LinearMap{T}, b::AbstractVector{T}; 
               x₀ = nothing,
               maxiter=min(100, size(operator, 1)),
               restart=min(20, size(operator, 1)),
               tol=1e-6, verbose=0, debug=false) where {T}

    n_size = length(b)

    # counters
    numrestarts::Int64 = 0
    restart_inds = Int64[]
    MvTime = Float64[]

    # initialization: Arnoldi basis and Hessenberg matrix
    V = zeros(n_size, restart + 1)
    H = zeros(restart + 1, restart)

    # residual history
    residuals = zeros(maxiter)
    denominatorvec = zeros(restart)
    denominator2 = 1.0

    # solution history
    if debug
        X = zeros(n_size, maxiter)
    end

    # if x0 is zero, set r0 = b
    if isnothing(x₀)
        β₀ = norm(b)
        V[:, 1] = b / β₀
    else
        # assert x₀ is a vector of the right size
        @assert length(x₀) == n_size
        push!(MvTime, @elapsed r₀ = operator * x₀)
        r₀ = b - r₀
        β₀ = norm(r₀)
        V[:, 1] = r₀ / β₀
    end

    i = 0 # iteration counter, set to 0 when restarted
    for j = 1:maxiter
        i += 1
        # Arnoldi iteration using MGS
        push!(MvTime, @elapsed w = operator * V[:, i])
        for j = 1:i
            H[j, i] = dot(V[:, j], w)
            w .-= H[j, i] * V[:, j]
        end
        H[i+1, i] = norm(w)
        V[:, i+1] = w / H[i+1, i]

        # update residual history
        if i == 1
            denominatorvec[1] = conj(-H[1, 1] / H[2, 1])
        else
            denominatorvec[i] = conj(-(H[1, i] + dot(denominatorvec[1:i-1], H[2:i, i])) / H[i+1, i])
        end
        denominator2 += abs2(denominatorvec[i])
        residuals[j] = β₀ / sqrt(denominator2)
        
        (verbose > 0) && println("restart = ", numrestarts + 1, ", iteration = ", i, ", res = ", residuals[j], "\n")

        happy = residuals[j] < tol
        needrestart = (i == restart)
        reachmaxiter = (j == maxiter)
        exitloop = (happy || reachmaxiter)

        solvels = (exitloop || needrestart || debug)
        if solvels
            # solve the Hessenberg least squares problem
            rhs = (UpperTriangular(H[2:i+1, 1:i]) \ denominatorvec[1:i]) * (-β₀ / denominator2)
            if isnothing(x₀)
                xᵢ = V[:, 1:i] * rhs
            else
                xᵢ = x₀ + V[:, 1:i] * rhs
            end

            debug && (X[:, j] = xᵢ)
            
            if exitloop
                if debug
                    return (; x=X[:, 1:j], residuals=residuals[1:j], MvTime=MvTime, restart_inds=restart_inds)
                else
                    return (; x=xᵢ, residuals=residuals[1:j], MvTime=MvTime, restart_inds=restart_inds)
                end
            end

            if needrestart
                i = 0
                x₀ = xᵢ
                push!(MvTime, @elapsed r₀ = operator * x₀)
                r₀ = b - r₀
                β₀ = norm(r₀)
                V[:, 1] = r₀ / β₀
                numrestarts += 1
                denominatorvec = zeros(maxiter)
                denominator2 = 1.0
                push!(restart_inds, j)
                (verbose > 0) && println("restarting: ", "iteration = ", j, ", r₀ = ", β₀, "\n")
            end
        end
    end
end
