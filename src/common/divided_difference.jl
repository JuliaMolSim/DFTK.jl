@doc raw"""
    divided_difference(f, fder, x, y)

Stable, AD-friendly first divided difference
```math
f[x, y] = \frac{f(x) - f(y)}{x - y},
```
which tends to ``f'(x)`` as ``y \to x``; `fder` is the derivative of `f`.

The direct formula `(f(x)-f(y))/(x-y)` is accurate to `ε/|x-y|` (catastrophic cancellation as
`x → y`) and is `0/0` at `x == y`. For `|x-y|` small we instead use the derivative midpoint
`(f'(x)+f'(y))/2`, which is accurate to `|x-y|²`. Balancing the two errors, we switch at
`|x-y| = ∛ε`, giving an overall accuracy `ε^{2/3}` and a finite, differentiable result through
`x == y`.

When `f(0) == 0`, `divided_difference(f, fder, x, 0)` is a stable way to evaluate `f(x)/x`,
including its `x → 0` limit `f'(0)` — the recurring pattern for the `1/r`-type singularities of
the interaction kernels in `coulomb.jl`. Pass a well-conditioned `f` (e.g. `x -> -expm1(-x)`
rather than `x -> 1 - exp(-x)`) so that the direct branch does not itself cancel.
"""
function divided_difference(f, fder, x::T, y) where {T}
    abs(x - y) < cbrt(eps(T)) && return (fder(x) + fder(y)) / 2
    (f(x) - f(y)) / (x - y)
end
