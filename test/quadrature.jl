@testitem "Simpson degress of exactness" begin
    import DFTK: simpson, simpson_nonuniform

    # Uniform, odd number of points -> degree of exactness 3
    xs = range(0, 1, 21)
    @test simpson((_, x) -> x^3, xs) ≈ 1/4 atol=1e-15

    # Uniform, even number of points -> degree of exactness 2
    xs = range(0, 1, 20)
    @test simpson((_, x) -> x^2, xs) ≈ 1/3 atol=1e-15

    # Non-uniform, odd number of points -> degree of exactness 2
    xs = range(0, 1, 21) .^ 2
    @test simpson((_, x) -> x^2, xs) ≈ 1/3 atol=1e-15

    # Non-uniform, even number of points -> degree of exactness 2
    xs = range(0, 1, 20) .^ 2
    @test simpson((_, x) -> x^2, xs) ≈ 1/3 atol=1e-15
end