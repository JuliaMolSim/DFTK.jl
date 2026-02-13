@testitem "CG" begin
    using DFTK
    using LinearAlgebra

    function test_cg(T)
        @testset "Test conjugate gradient method for $T" begin
            n = 10
            A = rand(Complex{T}, n, n)
            A = A' * A + I
            function op!(Ax, x)
                mul!(Ax, A, x)
            end
            b = rand(Complex{T}, n)
            tol = 1e-10 * (T == Float64) + 1e-5 * (T == Float32)

            res = DFTK.cg(op!, b; tol, maxiter=2n)

            # test convergence
            @test norm(A*res.x - b) ≤ 2tol
            @test res.converged

            # test type stability
            f(b) = DFTK.cg(op!, b; tol, maxiter=2n).x
            g(b) = DFTK.cg(op!, b; tol, maxiter=2n).residual_norms[1]
            @test res.x ≈ @inferred f(b)
            @test tol ≥ @inferred g(b)
        end
    end

    test_cg(Float32)
    test_cg(Float64)
end
