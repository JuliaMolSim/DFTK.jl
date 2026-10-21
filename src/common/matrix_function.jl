# Functions of Hermitian matrices and their derivatives. This is the small subset of
# MatrixFuns.jl (https://github.com/xuequan818/MatrixFuns.jl) needed here (Löwdin
# orthogonalization), kept in DFTK to avoid the dependency.

"""
`f(A) = V f(Λ) V'` for the Hermitian matrix `A = V Λ V'` with eigendecomposition `F`.
"""
matrix_function(f, F::Eigen) = F.vectors * Diagonal(f.(F.values)) * F.vectors'
matrix_function(f, A::Hermitian) = matrix_function(f, eigen(A))
# A^{-1/2} needs a well-conditioned A, e.g. an overlap of linearly independent orbitals
function matrix_function(::typeof(invsqrt), F::Eigen)
    @assert minimum(abs, F.values) > eps(real(eltype(F.values))) * maximum(abs, F.values)
    F.vectors * Diagonal(invsqrt.(F.values)) * F.vectors'
end

"""
Derivative of `f(A)` along the Hermitian perturbation `δA`, for the Hermitian matrix
`A = V Λ V'` with eigendecomposition `F` (Daleckii-Krein formula):

    V ((V' δA V) .* K) V',   K[i, j] = (f(λᵢ) - f(λⱼ)) / (λᵢ - λⱼ)

The eigendecomposition is passed so that it can be reused for several `δA`.
"""
function matrix_function_derivative(f, F::Eigen, δA)
    K = [divided_difference(f, λi, λj) for λi in F.values, λj in F.values]
    F.vectors * ((F.vectors' * δA * F.vectors) .* K) * F.vectors'
end
