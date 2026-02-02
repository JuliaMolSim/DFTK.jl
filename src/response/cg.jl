using LinearMaps
using LinearAlgebra: dot

function default_cg_print(info)
    @printf("%3d\t%1.2e\n", info.n_iter, info.residual_norm)
end

"""
Implementation of the conjugate gradient method which allows for preconditioning
and projection operations along iterations. The solver is optimized for generic
input arrays with multiple columns, although vector inputs are also supported.
To reduce allocations, operator A! and projector proj! are expected to be in-place
functions. A basic locking meachnism for converged columns is implemented.
"""
function cg!(x::AbstractArray{T}, A!, b::AbstractArray{T};
             precon=I, proj! =copy!, callback=identity,
             tol=1e-10*ones(T, size(b, 2)), maxiter=100, miniter=1,
             my_dots=nothing) where {T}

    # Columnwise dot products and norms, possibly custom
    if isnothing(my_dots)
        my_dots = columnwise_dots
        my_norms = columnwise_norms
    else
        my_norms(x) = sqrt.(real.(my_dots(x, x)))
    end

    # Utility to accept operators that do not implement active ranges for locking. In such
    # cases, all columns are always considered. Allows to use the same cg! implementation
    # across the board, without extra complication. For performance, when x and b are not
    # single column vectors, active ranges should be implemented.
    op_supports_active = false
    try
        A!(zeros_like(b), b; active=1:size(b, 2))
        op_supports_active = true
    catch MethodError
        op_supports_active = false
    end
    function apply_op!(Ax, x; active=nothing)
        if op_supports_active
            A!(Ax, x; active)
        else
            A!(Ax, x)
        end
    end

    # initialisation
    # r = b - Ax is the residual
    r = copy(b)
    # c is an intermediate variable to store A*p and precon\r
    c = zeros_like(b)

    # projection buffer to avoid allocations
    proj_buffer = similar(b)

    # save one matrix-vector product
    if !iszero(x)
        apply_op!(c, x)
        r .-= c
    end
    ldiv!(c, precon, r)
    γ = my_dots(r, c)
    # p is the descent direction
    p = copy(c)
    n_iter = 0
    residual_norms = zeros(real(T), size(b, 2)) # explicit typing for output type inference
    residual_norms .= to_cpu(my_norms(r))
    converged_cols = falses(size(b, 2))

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
        info = (; A!, b=full_b, n_iter, x=full_x, r=full_r, residual_norms=full_residuals,
                  converged, stage=:iterate)
        callback(info)
        n_iter += 1
        converged_cols .= full_residuals .<= tol
        if (n_iter ≥ miniter) && all(converged_cols)
            converged = true
            break
        end

        # Lock columns that are already converged. Because we can only take views with
        # contiguous ranges in GPU arrays, we lock the first and last columns.
        active = 1:size(b, 2)
        if op_supports_active
            locked_lb = findfirst(!, converged_cols)
            locked_ub = findlast(!, converged_cols)
            active = locked_lb : locked_ub
        end

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

        apply_op!(c, p; active)
        α = γ ./ my_dots(p, c)

        # update iterate and residual while ensuring they stay in Ran(proj)
        proj_buffer .= x .+ p .* α'
        proj!(x, proj_buffer)
        proj_buffer .= r .- c .* α'
        proj!(r, proj_buffer)
        residual_norms .= to_cpu(my_norms(r))

        # apply preconditioner and prepare next iteration. Preconditioner applied to full arrays,
        # because the FunctionPreconditioner type makes it hard to change the active set (cheap enough).
        ldiv!(full_c, precon, full_r)
        γ_prev = copy(γ)
        γ .= my_dots(r, c)
        β = γ ./ γ_prev
        proj_buffer .= c .+ p .* β'
        proj!(p, proj_buffer)
    end
    info = (; x=full_x, converged, tol, residual_norms=full_residuals, n_iter, maxiter, stage=:finalize)
    callback(info)
    info
end
cg(A, b::AbstractArray; kwargs...) = cg!(zeros_like(b), A, b; kwargs...)