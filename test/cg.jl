@testitem "CG" begin
    using DFTK
    using LinearAlgebra

    function test_cg(T)
        @testset "Test conjugate gradient method for $T" begin
            n = 10
            A = rand(Complex{T}, n, n)
            A = A' * A + I
            b = rand(Complex{T}, n)
            tol = 1e-10 * (T == Float64) + 1e-5 * (T == Float32)

            res = DFTK.cg(A, b; tol, maxiter=2n)

            # test convergence
            @test norm(A*res.x - b) ≤ 2tol
            @test res.converged

            # test type stability
            f(b) = DFTK.cg(A, b; tol, maxiter=2n).x
            g(b) = DFTK.cg(A, b; tol, maxiter=2n).residual_norm
            @test res.x ≈ @inferred f(b)
            @test tol ≥ @inferred g(b)
        end
    end

    test_cg(Float32)
    test_cg(Float64)

    @testset "Test CG with custom dot product" begin
        n = 10
        A = rand(ComplexF64, n, n)
        A = A' * A + I
        b = rand(ComplexF64, n)
        tol = 1e-10

        # Test with custom dot product (should give same result as default)
        custom_dot(x, y) = LinearAlgebra.dot(x, y)
        res1 = DFTK.cg(A, b; tol, maxiter=2n)
        res2 = DFTK.cg(A, b; tol, maxiter=2n, dot=custom_dot)

        @test res1.converged
        @test res2.converged
        @test res1.x ≈ res2.x
        @test norm(A*res2.x - b) ≤ 2tol
    end
end
