# Implementation of the preconditioned conjugate gradient method, which allows
# for projection operations, in order to solve linear systems in fixed subspace
# of the whole space.

"""
Implementation of the conjugate gradient method which allows for preconditioning
and projection operations along iterations.
"""
function CG!(x, A, b; precon=I, proj=ϕ->ϕ,
             tol=1e-10, maxiter=100, miniter=1, verbose=false)

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
    niter = 0

    # convergence history
    isconverged = false
    residual_history = []

    # preconditioned conjugate gradient
    while niter < maxiter
        niter += 1
        mul!(c, A, p)
        α = γ / dot(p, c)

        # update iterate and residual while ensuring they stay in Ran(proj)
        x .= proj(x .+ α .* p)
        r .= proj(r .- α .* c)
        resnorm = norm(r)

        # output
        verbose && @printf("%3d\t%1.2e\n", niter, resnorm)
        push!(residual_history, resnorm)
        if (niter > miniter) && resnorm <= tol
            isconverged = true
            break
        end

        # apply preconditioner and prepare next iteration
        ldiv!(c, precon, r)
        γ_prev, γ = γ, dot(r, c)
        β = γ / γ_prev
        p .= proj(c .+ β .* p)
    end

    ch = (; isconverged, tol, residual_history, niter)
    (x, ch)
end
CG(A, b; kwargs...) = CG!(zero(b), A, b; kwargs...)
