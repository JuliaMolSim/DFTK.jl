using DFTK
using Test
using IterativeSolvers
using LinearAlgebra: norm

@testset "Test conjugate gradient method convergence" begin
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
end

@testset "Test conjugate gradient method type stability" begin
    T = Float32
    n = 10
    A = rand(Complex{T}, n, n)
    A = A' * A + I
    b = rand(Complex{T}, n)
    tol = 1e-5
    x = A \ b

    f(b) = DFTK.cg(A, b; tol, maxiter=2n).x
    g(b) = DFTK.cg(A, b; tol, maxiter=2n).residual_norm
    h(b) = eltype(DFTK.cg(A, b; tol, maxiter=2n).residual_history)

    @test x ≈ @inferred f(b)
    @test tol ≥ @inferred g(b)
    @test T == @inferred h(b)
end
