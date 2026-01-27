using LinearMaps
using LinearAlgebra: dot

function default_cg_print(info)
    @printf("%3d\t%1.2e\n", info.n_iter, info.residual_norm)
end

"""
Implementation of the conjugate gradient method which allows for preconditioning
and projection operations along iterations.
"""
function cg!(x::AbstractVector{T}, A::LinearMap{T}, b::AbstractVector{T};
             precon=I, proj=identity, callback=identity,
             tol=1e-10, maxiter=100, miniter=1,
             my_dot=dot) where {T}
    my_norm(x) = sqrt(real(my_dot(x, x)))

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
    γ = my_dot(r, c)
    # p is the descent direction
    p = copy(c)
    n_iter = 0
    residual_norm = my_norm(r)

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
        α = γ / my_dot(p, c)

        # update iterate and residual while ensuring they stay in Ran(proj)
        x .= proj(x .+ α .* p)
        r .= proj(r .- α .* c)
        residual_norm = my_norm(r)

        # apply preconditioner and prepare next iteration
        ldiv!(c, precon, r)
        γ_prev, γ = γ, my_dot(r, c)
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

# TEST TODO: generalize to a single cg! implementation. Prob need a new struct for A
function cg!(x::AbstractArray{T}, A!, b::AbstractArray{T};
             precon=I, proj! =identity, callback=identity,
             tol=1e-10*ones(T, size(b, 2)), maxiter=100, miniter=1) where {T}

    # initialisation
    # r = b - Ax is the residual
    r = copy(b)
    # c is an intermediate variable to store A*p and precon\r
    c = zeros_like(b)

    # projection buffer to avoid allocations
    proj_buffer = similar(b)

    # save one matrix-vector product
    if !iszero(x)
        A!(c, x)
        r .-= c
    end
    ldiv!(c, precon, r)
    γ = columnwise_dots(r, c)
    # p is the descent direction
    p = copy(c)
    n_iter = 0
    residual_norms = columnwise_norms(r)
    # move tols to GPU for efficiency
    col_tols = oftype(residual_norms, tol)

    # convergence history
    converged = false

    # Keep track of full arrays to enable locking of converged columns
    full_x = x
    full_r = r
    full_c = c
    full_p = p
    full_b = b
    full_buff = proj_buffer
    full_γ = γ
    full_residuals = residual_norms

    # preconditioned conjugate gradient
    while n_iter < maxiter
        # output
        info = (; A!, full_b, n_iter, full_x, full_r, full_residuals, converged, stage=:iterate)
        callback(info)
        n_iter += 1
        converged_cols = full_residuals .<= col_tols
        if (n_iter ≥ miniter) && all(converged_cols)
            converged = true
            break
        end

        # Lock columns that are already converged. Because we can only take views with
        # contiguous ranges in GPU arrays, we lock the first and last columns.
        locked_lb = findfirst(!, converged_cols)
        locked_ub = findlast(!, converged_cols)
        active = locked_lb : locked_ub

        @views begin
            x = full_x[:, active]
            r = full_r[:, active]
            c = full_c[:, active]
            p = full_p[:, active]
            b = full_b[:, active]
            proj_buffer = full_buff[:, active]
            γ = full_γ[active]
            residual_norms = full_residuals[active]
        end

        A!(c, p; active)
        α = γ ./ columnwise_dots(p, c)

        # update iterate and residual while ensuring they stay in Ran(proj)
        proj_buffer .= x .+ p .* α'
        proj!(x, proj_buffer)
        proj_buffer .= r .- c .* α'
        proj!(r, proj_buffer)
        residual_norms .= columnwise_norms(r)

        # apply preconditioner and prepare next iteration. Preconditioner applied to full arrays,
        # because the FunctionPreconditioner type makes it hard to change the active set (cheap enough).
        ldiv!(full_c, precon, full_r)
        γ_prev = copy(γ)
        γ .= columnwise_dots(r, c)
        β = γ ./ γ_prev
        proj_buffer .= c .+ p .* β'
        proj!(p, proj_buffer)
    end
    info = (; x=full_x, converged, tol, residual_norms=to_cpu(full_residuals), n_iter, maxiter, stage=:finalize)
    callback(info)
    info
end
cg(A, b::AbstractArray; kwargs...) = cg!(zeros_like(b), A, b; kwargs...)