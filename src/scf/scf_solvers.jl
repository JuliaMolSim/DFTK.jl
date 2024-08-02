# these provide fixed-point solvers that can be passed to self_consistent_field()

# the fp_solver function must accept being called like
# fp_solver(f, x0, info0, tol, maxiter), where f(x, info) is the fixed-point map. 
# It must return an object supporting res.fixpoint, res.info and res.converged

"""
Create a damped SCF solver updating the density as
`x = β * x_new + (1 - β) * x`
"""
function scf_damping_solver(β=0.2)
    function fp_solver(f, x0, info0, maxiter; tol=1e-6)
        β = convert(eltype(x0), β)
        converged = false
        x = copy(x0)
        info = info0
        for i in 1:maxiter
            x_new, info = f(x, info)

            if norm(x_new - x) < tol
                x = x_new
                converged = true
                break
            end

            x = @. β * x_new + (1 - β) * x
        end
        (; fixpoint=x, info, converged)
    end
    fp_solver
end

"""
Create a simple anderson-accelerated SCF solver. `m` specifies the number
of steps to keep the history of.
"""
function scf_anderson_solver(m=10; kwargs...)
    function anderson(f, x0, info0, maxiter; tol=1e-6)
        T = eltype(x0)
        x = x0
        info = info0

        converged = false
        acceleration = AndersonAcceleration(; m, kwargs...)
        for n = 1:maxiter
            fx, info = f(x, info)
            residual = fx - x
            converged = norm(residual) < tol
            converged && break
            x = acceleration(x, one(T), residual)
        end
        (; fixpoint=x, info, converged)
    end
end
