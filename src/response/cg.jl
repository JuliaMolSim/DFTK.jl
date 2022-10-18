using LinearMaps

"""
Implementation of the conjugate gradient method which allows for preconditioning
and projection operations along iterations.
"""
function cg!(x::AbstractVector{T}, A::LinearMap{T}, b::AbstractVector{T};
             precon=I, proj=identity, callback=info->nothing,
             tol=1e-10, maxiter=100, miniter=1, verbose=false) where {T}

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
    γ = dot(r, c)
    # p is the descent direction
    p = copy(c)
    n_iter = 0
    residual_norm = zero(real(T))

    # convergence history
    converged = false
    residual_history = real(T)[]

    # preconditioned conjugate gradient
    while n_iter < maxiter
        n_iter += 1
        mul!(c, A, p)
        α = γ / dot(p, c)

        # update iterate and residual while ensuring they stay in Ran(proj)
        x .= proj(x .+ α .* p)
        r .= proj(r .- α .* c)
        residual_norm = norm(r)

        # output
        verbose && @printf("%3d\t%1.2e\n", n_iter, residual_norm)
        push!(residual_history, residual_norm)
        info = (; n_iter, x, r, A, b)
        callback(info)
        if (n_iter > miniter) && residual_norm <= tol
            converged = true
            break
        end

        # apply preconditioner and prepare next iteration
        ldiv!(c, precon, r)
        γ_prev, γ = γ, dot(r, c)
        β = γ / γ_prev
        p .= proj(c .+ β .* p)
    end

    (; x, converged, tol, residual_norm, residual_history,
     iterations=n_iter, maxiter)
end
cg(x::AbstractVector, A::AbstractMatrix, b::AbstractVector; kwargs...) = cg!(x, LinearMap(A), b; kwargs...)
cg(A::LinearMap, b::AbstractVector; kwargs...) = cg!(zero(b), A, b; kwargs...)
cg(A::AbstractMatrix, b::AbstractVector; kwargs...) = cg(LinearMap(A), b; kwargs...)
