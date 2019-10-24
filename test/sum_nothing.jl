using Test
using DFTK: sum_nothing

@testset "Test sum_nothing" begin
    @test sum_nothing() === nothing
    @test sum_nothing(nothing, nothing) === nothing
    @test sum_nothing(nothing, nothing, nothing) === nothing

    @test sum_nothing(1, nothing) === 1
    @test sum_nothing(nothing, 1) === 1
    @test sum_nothing(1, nothing, 2) === 3
    @test sum_nothing(1, nothing, 2, nothing, nothing) === 3
    @test sum_nothing(1, 3, 2, nothing, nothing) === 6
end
