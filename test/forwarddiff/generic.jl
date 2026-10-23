### Generic ForwardDiff tests, independent of architecture

@testitem "Derivative of complex function" #=
    =#    tags=[:dont_test_mpi, :minimal] setup=[ForwardDiffWrappers] begin
    using DFTK
    using ForwardDiff
    using LinearAlgebra
    using SpecialFunctions
    using FiniteDifferences

    α = randn(ComplexF64)
    erfcα = x -> erfc(α * x)

    x0  = randn()
    fd1 = ForwardDiffWrappers.tagged_derivative(erfcα, x0)
    fd2 = FiniteDifferences.central_fdm(5, 1)(erfcα, x0)
    @test norm(fd1 - fd2) < 1e-8
end

@testitem "Higher derivatives of Fermi-Dirac occupation" #=
    =#    tags=[:dont_test_mpi, :minimal] begin
    using DFTK
    using ForwardDiff

    smearing = Smearing.FermiDirac()
    f(x) = Smearing.occupation(smearing, x)

    function compute_nth_derivative(n, f, x)
        (n == 0) && return f(x)
        ForwardDiff.derivative(x -> compute_nth_derivative(n - 1, f, x), x)
    end

    @testset "Avoid NaN from exp-overflow for large x" begin
        T = Float64
        x = log(floatmax(T)) / 2 + 1
        for n in 0:8
            @testset "Derivative order $n" begin
                y = compute_nth_derivative(n, f, x)
                @test isfinite(y)
                @test abs(y) ≤ eps(T)
            end
        end
    end
end

@testitem "Derivative of nonlocal projectors at [0,0,0]" #=
    =#    tags=[:dont_test_mpi, :minimal] setup=[TestCases] begin
    using DFTK
    using ForwardDiff
    using StaticArrays
    using .TestCases: pd_lda_family

    # has projectors up to l=3
    psp = load_psp(pd_lda_family[:Hg])
    projs_by_l = [1:2, 3:8, 9:18, 19:25]

    for l in 0:3, α in 1:3
        @testset "l = $l, α = $α" begin
            p = @SVector[0, 0, 0]
            dp = @SVector[float(i == α) for i in 1:3]

            f(ε) = DFTK.build_projector_form_factors(psp, [p+ε*dp])[projs_by_l[l+1]]

            ff_ad = ForwardDiff.derivative(f, 0.0)

            # High l are sensitive to numerical noise, so we only use FD for l=1,
            # and we know that the derivative should be zero for l=0,2,3
            ff_ref = zero(ff_ad)
            if l == 1
                h = 1e-4
                ff_ref .= (-3*f(0.0) + 4*f(h) - f(2h)) / 2h
            end
            @test ff_ad ≈ ff_ref rtol=1e-7 atol=1e-10
        end
    end
end
@testitem "Matrix products with a Dual scaling factor" #=
    =#    tags=[:dont_test_mpi, :minimal] setup=[ForwardDiffWrappers] begin
    using DFTK
    using ForwardDiff
    using LinearAlgebra
    using Random

    # LinearAlgebra turns γ * A * B into mul!(C, A, B, γ, false), so the scaling factor of
    # the BLAS-dispatched Dual matrix product can itself be a Dual.
    for T in (Float64, ComplexF64)
        A = randn(Xoshiro(1), T, 4, 3)
        B = randn(Xoshiro(2), T, 3, 5)
        f(t) = real(sum((1 + t) * (A .* (1 + 2t)) * B))  # = (1 + t)(1 + 2t) sum(A * B)
        g(t) = real(sum((1 + t) * (A .* (1 + 2t)) * (B .* (1 - t))))
        @test ForwardDiffWrappers.tagged_derivative(f, 0.0) ≈ 3 * real(sum(A * B))
        @test ForwardDiffWrappers.tagged_derivative(g, 0.0) ≈ 2 * real(sum(A * B))
    end
end
@testitem "Derivative of matrix functions with degenerate eigenvalues" #=
    =#    tags=[:dont_test_mpi, :minimal] setup=[ForwardDiffWrappers] begin
    using DFTK
    using DFTK: matrix_function, invsqrt
    using LinearAlgebra

    # Eigenvalues (2, 2, 3) in a random basis. ForwardDiff's eigen rule divides by
    # eigenvalue differences and gives NaN here; the Dual overload of matrix_function
    # must not.
    U = Matrix(qr(randn(ComplexF64, 3, 3)).Q)
    A = U * Diagonal([2.0, 2.0, 3.0]) * U'
    δA = Hermitian(randn(ComplexF64, 3, 3))
    # invsqrt uses its exact divided difference, exp the generic one from ForwardDiff
    for f in (invsqrt, exp)
        fA(t) = matrix_function(f, Hermitian(A + t * δA))
        h = 1e-5
        reference = (fA(h) - fA(-h)) / 2h
        @test ForwardDiffWrappers.tagged_derivative(fA, 0.0) ≈ reference rtol=1e-7
    end
end

@testitem "Löwdin orthogonalization ForwardDiff rule" #=
    =#    tags=[:minimal] setup=[ForwardDiffWrappers] begin
    using DFTK
    using LinearAlgebra

    φ_orthonormal = Matrix(qr(randn(ComplexF64, 8, 3)).Q)  # case 1: φ'φ = I
    φ_generic     = randn(ComplexF64, 8, 3)                # case 2: φ'φ ≠ I
    δφ = randn(ComplexF64, 8, 3)
    for φ in (φ_orthonormal, φ_generic)
        P(t) = DFTK.ortho_lowdin(φ + t * δφ)
        h = 1e-5
        reference = (P(h) - P(-h)) / 2h
        @test ForwardDiffWrappers.tagged_derivative(P, 0.0) ≈ reference rtol=1e-7
    end
end
