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
Create a simple fixed-point iterations-based solver, updating the density
as -`x = damping * x_new + (1 - damping) * x`. For applying damping or mixing,
see also the other keyword arguments of [`self_consistent_field`](@ref).
"""
function scf_damping_solver(; damping=1.0)
    function fp_solver(f, x0, info0; maxiter)
        β = convert(eltype(x0), damping)
        x = x0
        info = info0
        for i = 1:maxiter
            fx, info = f(x, info)
            if info.converged || info.timedout
                break
            end
            x = @. β * fx + (1 - β) * x
        end
        (; fixpoint=x, info)
    end
end

@doc raw"""
Create an anderson-accelerated SCF solver for the [`self_consistent_field`](@ref) solver.

## Keyword arguments
- `m::Integer`       (default: `10`) Maximal Anderson history size
- `m_start::Integer` (default: `1`)  Start collecting history in the `m_start`th SCF iteration
- `maxcond::Real` (default: `1e6`)
  Maximal condition number in Anderson matrix; a larger value triggers truncation of the
  older entries in the Anderson history.
- `errorfactor::Real` (default: `1e5`): We follow [^CDLS21] (adaptive Anderson acceleration)
  and drop iterates, which do not satisfy
  ```math
      \|r(ρᵢ)\| < \text{errorfactor} minᵢ \|r(ρᵢ)\|
  ```
  where $r(ρ)$ denotes the preconditioned SCF residual corresponding to density $ρ$.
  This means the best way to save memory is to reduce `errorfactor` to `1e3` or `100`,
  which reduces the effective history size.

[^CDLS21]: Chupin, Dupuy, Legendre, Séré. Math. Model. Num. Anal. **55**, 2785 (2021) dDOI [10.1051/m2an/2021069](https://doi.org/10.1051/m2an/2021069)
"""
function scf_anderson_solver(; m_start::Integer=1, kwargs...)
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
            if i < m_start
                @debug "Skipping Anderson acceleration in iteration $i"
                x = fx
            else
                @debug "Using Anderson acceleration in iteration $i"
                residual = fx - x
                x = acceleration(x, one(T), residual)
            end
        end
        (; fixpoint=x, info)
    end
end
