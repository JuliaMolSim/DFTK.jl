# Functions of Hermitian matrices and their derivatives. This is the small subset of
# MatrixFuns.jl (https://github.com/xuequan818/MatrixFuns.jl) needed here (Löwdin
# orthogonalization), kept in DFTK to avoid the dependency.

"""
`f(A) = V f(Λ) V'` for the Hermitian matrix `A = V Λ V'` with eigendecomposition `F`.
"""
matrix_function(f, F::Eigen) = F.vectors * Diagonal(f.(F.values)) * F.vectors'
matrix_function(f, A::Hermitian) = matrix_function(f, eigen(A))

"""
Derivative of `f(A)` along the Hermitian perturbation `δA`, for the Hermitian matrix
`A = V Λ V'` with eigendecomposition `F` (Daleckii-Krein formula):

    V ((V' δA V) .* K) V',   K[i, j] = (f(λᵢ) - f(λⱼ)) / (λᵢ - λⱼ)

The eigendecomposition is passed so that it can be reused for several `δA`.
"""
function matrix_function_derivative(f, F::Eigen, δA)
    K = divided_difference.(f, F.values, transpose(F.values))
    F.vectors * ((F.vectors' * δA * F.vectors) .* K) * F.vectors'
end

# The generic ForwardDiff rule for eigen(::Hermitian) divides by eigenvalue differences and
# is wrong for degenerate eigenvalues, e.g. of the overlap of atomic orbitals in a crystal.
# Nested Duals (higher derivatives) still go through that generic rule for the inner level.
function matrix_function(f, A::Hermitian{Complex{ForwardDiff.Dual{Tg,V,N}}}) where {Tg,V,N}
    F = eigen(Hermitian(ForwardDiff.value.(A)))
    δfA = ntuple(p -> matrix_function_derivative(f, F, ForwardDiff.partials.(A, p)), N)
    map(matrix_function(f, F), δfA...) do x, δx...
        Complex(ForwardDiff.Dual{Tg}(real(x), real.(δx)),
                ForwardDiff.Dual{Tg}(imag(x), imag.(δx)))
    end
end
