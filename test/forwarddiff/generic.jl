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