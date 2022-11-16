"""
Complex-analytic extension of `LinearAlgebra.norm(x)` from real to complex inputs.
Needed for phonons as we want to perform a matrix-vector product `f'(x)·h`, where `f` is
a real-to-real function and `h` a complex vector. To do this using automatic
differentiation, we can extend analytically f to accept complex inputs, then differentiate
`t -> f(x+t·h)`. This will fail if non-analytic functions like norm are used for complex
inputs, and therefore we have to redefine it.
"""
norm_cplx(x) = sqrt(sum(xx -> xx * xx, x))

"""
Compute the square of the ℓ²-norm for instances of our static structure Vec3.
"""
norm2(G::Vec3) = sum(abs2, G)
