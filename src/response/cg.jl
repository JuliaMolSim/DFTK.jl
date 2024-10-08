using LinearMaps

function default_cg_print(info)
    @printf("%3d\t%1.2e\n", info.n_iter, info.residual_norm)
end

"""
Implementation of the conjugate gradient method which allows for preconditioning
and projection operations along iterations.

To use this function with MPI:
- Make `A`, `precon` and `proj` MPI-aware.
- Pass a communicator to use for the `dot` and `norm` calls. By default, they do not communicate.
"""
function cg!(x::AbstractVector{T}, A::LinearMap{T}, b::AbstractVector{T};
             precon=I, proj=identity, callback=identity,
             tol=1e-10, maxiter=100, miniter=1,
             comm::MPI.Comm=MPI.COMM_SELF) where {T}

    mpi_dot(x1, x2) = mpi_sum(dot(x1, x2), comm)
    mpi_norm(x1) = sqrt(mpi_sum(norm2(x1), comm))

    # initialisation
    # r = b - Ax is the residual
    r = copy(b)
    # c is an intermediate variable to store A*p and precon\r
    c = zero(b)

    # save one matrix-vector product
    if !iszero(x)
        mul!(c, A, x)
        r .-= c
    end
    ldiv!(c, precon, r)
    γ = mpi_dot(r, c)
    # p is the descent direction
    p = copy(c)
    n_iter = 0
    residual_norm = mpi_norm(r)

    # convergence history
    converged = false

    # preconditioned conjugate gradient
    while n_iter < maxiter
        # output
        info = (; A, b, n_iter, x, r, residual_norm, converged, stage=:iterate)
        callback(info)
        n_iter += 1
        if (n_iter ≥ miniter) && residual_norm ≤ tol
            converged = true
            break
        end
        mul!(c, A, p)
        α = γ / mpi_dot(p, c)

        # update iterate and residual while ensuring they stay in Ran(proj)
        x .= proj(x .+ α .* p)
        r .= proj(r .- α .* c)
        residual_norm = mpi_norm(r)

        # apply preconditioner and prepare next iteration
        ldiv!(c, precon, r)
        γ_prev, γ = γ, mpi_dot(r, c)
        β = γ / γ_prev
        p .= proj(c .+ β .* p)
    end
    info = (; x, converged, tol, residual_norm, n_iter, maxiter, stage=:finalize)
    callback(info)
    info
end
cg!(x::AbstractVector, A::AbstractMatrix, b::AbstractVector; kwargs...) = cg!(x, LinearMap(A), b; kwargs...)
cg(A::LinearMap, b::AbstractVector; kwargs...) = cg!(zero(b), A, b; kwargs...)
cg(A::AbstractMatrix, b::AbstractVector; kwargs...) = cg(LinearMap(A), b; kwargs...)
