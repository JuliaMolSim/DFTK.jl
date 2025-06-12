using DFTK
using LinearAlgebra
using Test

@testitem "Inexact GMRES" begin
    using DFTK
    using LinearAlgebra

    struct MatNoisy{T}
        mat::Matrix{T}
    end
    function DFTK.mul_approximate(A::MatNoisy{T}, x; tol) where {T}
        y_exact = A.mat * x
        e = randn(T, length(y_exact))
        Ax = y_exact + (e / norm(e) * T(tol))
        (; Ax, info=(; tol))
    end
    Base.size(A::MatNoisy, args...) = size(A.mat, args...)

    function test_gmres(T; krylovdim=10, s=1, use_diagonal_preconditioner=false)
        @testset "Test GMRES tor type $T" begin
            n = 300
            A = rand(Complex{T}, n, n) + 10I
            b = rand(Complex{T}, n)
            tol = 1e-10 * (T == Float64) + 1e-5 * (T == Float32)

            Anoisy = MatNoisy(A)
            precon = use_diagonal_preconditioner ? Diagonal(A) : I
            callback = identity  # DFTK.default_gmres_print
            res = DFTK.inexact_gmres(Anoisy, b; tol, krylovdim, s, precon, callback)

            # Test convergence
            # Note: Currently our preconditioned GMRES ensures exactly this condition
            #       *including* the preconditioner in the norm
            @test norm(precon \ (A * res.x - b)) ≤ 2tol
            @test res.converged

            # Test type stability
            let tol=1e-5
                f(b) = DFTK.inexact_gmres(Anoisy, b; tol).x
                g(b) = DFTK.inexact_gmres(Anoisy, b; tol).residual_norm
                @test res.x ≈ (@inferred f(b)) atol=tol
                @test tol   ≥ @inferred g(b)
            end
        end
    end

    test_gmres(Float32)
    test_gmres(Float64)
    test_gmres(Float64; use_diagonal_preconditioner=true)
    test_gmres(Float64; s=1000, krylovdim=100)  # Trigger a restart by too large s
end
