using DFTK
using Test
using IterativeSolvers
using LinearAlgebra: norm

@testset "Test conjugate gradient method" begin
    n = 10
    A = rand(n, n)
    A = A' * A + I
    b = rand(n)
    tol = 1e-10

    x, ch = DFTK.CG(A, b; tol, maxiter=2n)

    @test norm(A*x - b) ≤ 1e-10
    @test ch.isconverged
    @test ch.niter == n+1
end
