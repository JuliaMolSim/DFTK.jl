# these provide fixed-point solvers that can be passed to `self_consistent_field`

# the fp_solver function must accept being called like
# `fp_solver(f, x0, info0; maxiter)`, where `f` is the fixed-point map.
#
# The fixed-point map `f` is expected to be called as such:
#    `(fx, info) = f(x, info)`
# The`info` contains auxiliary state, including two boolean flags:
#    `info.converged`: flagged inside `f` if the convergence criterion is achieved.
#    `info.timedout`: flagged inside `f` if a timeout is achieved.
# The fixed-point function `f` is just passive in the sense that it flags these,
# but still can be called further. The decision-making is left to the solver,
# with the default convention being that either of these flags leads to termination.
#
# The solver must return an object supporting res.fixpoint and res.info

"""
Create a damped SCF solver updating the density as
`x = damping * x_new + (1 - damping) * x`
"""
function scf_damping_solver(; damping=0.2)
    function fp_solver(f, x0, info0; maxiter)
        β = convert(eltype(x0), damping)
        x = copy(x0)
        info = info0
        for i = 1:maxiter
            x_new, info = f(x, info)
            if info.converged || info.timedout
                x = x_new
                break
            end
            x = @. β * x_new + (1 - β) * x
        end
        (; fixpoint=x, info)
    end
    fp_solver
end

"""
Create a simple anderson-accelerated SCF solver.
"""
function scf_anderson_solver(; kwargs...)
    function anderson(f, x0, info0; maxiter)
        T = eltype(x0)
        x = x0
        info = info0
        acceleration = AndersonAcceleration(; kwargs...)
        for i = 1:maxiter
            fx, info = f(x, info)
            if info.converged || info.timedout
                break
            end
            residual = fx - x
            x = acceleration(x, one(T), residual)
        end
        (; fixpoint=x, info)
    end
end
