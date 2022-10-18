using DFTK
using Test
using IterativeSolvers
using LinearAlgebra: norm

@testset "Test conjugate gradient method" begin
    T = Float64
    n = 10
    A = rand(Complex{T}, n, n)
    A = A' * A + I
    b = rand(Complex{T}, n)
    tol = 1e-10

    res = DFTK.cg(A, b; tol, maxiter=2n)

    @test norm(A*res.x - b) ≤ tol
    @test res.converged
    @test res.iterations == n+1
    @test typeof(res.residual_norm) == T
    @test eltype(res.residual_history) == T
end
