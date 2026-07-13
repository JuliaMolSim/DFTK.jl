@testitem "Simpson degress of exactness" begin
    import DFTK: simpson, simpson_nonuniform

    function assert_degree_of_exactness(rule, xs, degree)
        @test rule((_, x) -> x^degree, xs) ≈ 1/(degree+1) atol=1e-15
        @test abs(rule((_, x) -> x^(degree+1), xs) - 1/(degree+2)) > 1e-10
    end

    # Uniform, odd number of points -> degree of exactness 3
    xs = range(0, 1, 21)
    assert_degree_of_exactness(simpson, xs, 3)

    # Uniform, even number of points -> degree of exactness 2
    xs = range(0, 1, 20)
    assert_degree_of_exactness(simpson, xs, 2)

    # Non-uniform, odd number of points -> degree of exactness 2
    xs = range(0, 1, 21) .^ 2
    assert_degree_of_exactness(simpson_nonuniform, xs, 2)

    # Non-uniform, even number of points -> degree of exactness 2
    xs = range(0, 1, 20) .^ 2
    assert_degree_of_exactness(simpson_nonuniform, xs, 2)
end