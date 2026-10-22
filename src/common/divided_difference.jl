"""
`(f(x) - f(y))/(x - y)`, computed stably in the case where `x` and `y` are close.
Without `fder`, the derivative `f'` is computed by ForwardDiff. Functions with a closed form
that is exact for `x ≈ y` overload the method without `fder`, which can fall back to the
method with `fder` (e.g. `fermi_dirac`).
"""
function divided_difference(f, fder, x::T, y) where {T}
    # (f(x) - f(y))/(x - y) is accurate to ε/|x-y|
    # so for x ~= y we use the approximation (f'(x)+f'(y))/2,
    # which is accurate to |x-y|^2, and therefore better when |x-y| ≤ cbrt(ε)
    # The resulting method is accurate to ε^2/3
    abs(x-y) < cbrt(eps(T)) && return (fder(x) + fder(y))/2
    (f(x)-f(y)) / (x-y)
end
divided_difference(f, x, y) = divided_difference(f, z -> ForwardDiff.derivative(f, z), x, y)
